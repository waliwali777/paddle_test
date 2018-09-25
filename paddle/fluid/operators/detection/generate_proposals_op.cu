/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <stdio.h>
#include <string>
#include <vector>
#include "cub/cub.cuh"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/gather.cu.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

std::ostream &operator<<(std::ostream &os, const Tensor &t) {
  Tensor tt;
  framework::TensorCopySync(t, platform::CPUPlace(), &tt);

  os << "dim: " << t.dims() << " \n";

  int64_t size = t.numel();
  // int64_t size = 120;
  for (int64_t i = 0; i < size; ++i) {
    if (framework::IsType<float>(t.type())) {
      os << tt.data<float>()[i] << " ";
    } else if (framework::IsType<int64_t>(t.type())) {
      os << tt.data<int64_t>()[i] << " ";
    } else if (framework::IsType<int>(t.type())) {
      os << tt.data<int>()[i] << " ";
    } else {
      PADDLE_THROW("LoDTensor data type not in [float, int64_t]");
    }
  }
  return os;
}

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y)-1) / (y))
#endif

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

// const float kBBoxClipDefault = std::log(1000.0 / 16.0);
// const int kThreadsPerBlock = sizeof(uint64_t) * 8;

// #define kBBoxClipDefault std::log(1000.0 / 16.0)
// #define kBBoxClipDefault 1.79588
#define kThreadsPerBlock sizeof(uint64_t) * 8

template <typename T>
__global__ void RangeInitKernel(const T start, const T delta, const int size,
                                T *out) {
  CUDA_1D_KERNEL_LOOP(i, size) { out[i] = start + i * delta; }
}

template <typename T>
void SortDescending(const platform::CUDADeviceContext &ctx, const Tensor &value,
                    Tensor *value_out, Tensor *index_out) {
  int num = value.numel();
  Tensor index_in_t;
  int *idx_in = index_in_t.mutable_data<int>({num}, ctx.GetPlace());
  int block = 512;
  auto stream = ctx.stream();
  RangeInitKernel<<<DIVUP(num, block), block, 0, stream>>>(0, 1, num, idx_in);
  int *idx_out = index_out->mutable_data<int>({num}, ctx.GetPlace());

  const T *keys_in = value.data<T>();
  T *keys_out = value_out->mutable_data<T>({num}, ctx.GetPlace());

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending<T, int>(
      d_temp_storage, temp_storage_bytes, keys_in, keys_out, idx_in, idx_out,
      num);

  // Allocate temporary storage
  auto place = boost::get<platform::CUDAPlace>(ctx.GetPlace());
  d_temp_storage = memory::Alloc(place, temp_storage_bytes);

  // Run sorting operation
  cub::DeviceRadixSort::SortPairsDescending<T, int>(
      d_temp_storage, temp_storage_bytes, keys_in, keys_out, idx_in, idx_out,
      num);

  memory::Free(place, d_temp_storage);
}

template <typename T>
__device__ __forceinline__ T Min(T x, T y) {
  return x < y ? x : y;
}

template <typename T>
__device__ __forceinline__ T Max(T x, T y) {
  return x > y ? x : y;
}

template <typename T>
__global__ void BoxDecodeAndClipKernel(const T *anchor, const T *deltas,
                                       const T *var, const int *index,
                                       const T *im_info, const int num,
                                       T *proposals) {
  T kBBoxClipDefault = log(1000.0 / 16.0);
  CUDA_1D_KERNEL_LOOP(i, num) {
    int k = index[i] * 4;
    T axmin = anchor[k];
    T aymin = anchor[k + 1];
    T axmax = anchor[k + 2];
    T aymax = anchor[k + 3];

    T w = axmax - axmin + 1.0;
    T h = aymax - aymin + 1.0;
    T cx = axmin + 0.5 * w;
    T cy = aymin + 0.5 * h;

    T dxmin = deltas[k];
    T dymin = deltas[k + 1];
    T dxmax = deltas[k + 2];
    T dymax = deltas[k + 3];

    T d_cx = 0., d_cy = 0., d_w = 0., d_h = 0.;
    if (var) {
      d_cx = cx + dxmin * w * var[k];
      d_cy = cy + dymin * h * var[k + 1];
      d_w = exp(Min<T>(dxmax * var[k + 2], kBBoxClipDefault)) * w;
      d_h = exp(Min<T>(dymax * var[k + 3], kBBoxClipDefault)) * h;
    } else {
      d_cx = cx + dxmin * w;
      d_cy = cy + dymin * h;
      d_w = exp(Min<T>(dxmax, kBBoxClipDefault)) * w;
      d_h = exp(Min<T>(dymax, kBBoxClipDefault)) * h;
    }

    T oxmin = d_cx - d_w * 0.5;
    T oymin = d_cy - d_h * 0.5;
    T oxmax = d_cx + d_w * 0.5 - 1.;
    T oymax = d_cy + d_h * 0.5 - 1.;

    proposals[i * 4] = Max<T>(Min<T>(oxmin, im_info[1] - 1.), 0.);
    proposals[i * 4 + 1] = Max<T>(Min<T>(oymin, im_info[0] - 1.), 0.);
    proposals[i * 4 + 2] = Max<T>(Min<T>(oxmax, im_info[1] - 1.), 0.);
    proposals[i * 4 + 3] = Max<T>(Min<T>(oymax, im_info[0] - 1.), 0.);
  }
}

template <typename T, int BlockSize>
__global__ void FilterBBoxes(const T *bboxes, const T *im_info,
                             const T min_size, const int num, int *keep_num,
                             int *keep) {
  T im_h = im_info[0];
  T im_w = im_info[1];
  T im_scale = im_info[2];

  int cnt = 0;
  __shared__ int keep_index[BlockSize];

  CUDA_1D_KERNEL_LOOP(i, num) {
    keep_index[threadIdx.x] = -1;
    __syncthreads();

    int k = i * 4;
    T xmin = bboxes[k];
    T ymin = bboxes[k + 1];
    T xmax = bboxes[k + 2];
    T ymax = bboxes[k + 3];

    T w = xmax - xmin + 1.0;
    T h = ymax - ymin + 1.0;
    T cx = xmin + w / 2.;
    T cy = ymin + h / 2.;

    T w_s = (xmax - xmin) / im_scale + 1.;
    T h_s = (ymax - ymin) / im_scale + 1.;

    if (w_s >= min_size && h_s >= min_size && cx <= im_w && cy <= im_h) {
      keep_index[threadIdx.x] = i;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int size = (num - i) < BlockSize ? num - i : BlockSize;
      for (int j = 0; j < size; ++j) {
        if (keep_index[j] > -1) {
          keep[cnt++] = keep_index[j];
        }
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    keep_num[0] = cnt;
  }
}

__device__ inline float IoU(float const *const a, float const *const b) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

__global__ void NMSKernel(const int n_boxes, const float nms_overlap_thresh,
                          const float *boxes, uint64_t *mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  const int row_size =
      min(n_boxes - row_start * kThreadsPerBlock, kThreadsPerBlock);
  const int col_size =
      min(n_boxes - col_start * kThreadsPerBlock, kThreadsPerBlock);

  __shared__ float block_boxes[kThreadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        boxes[(kThreadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        boxes[(kThreadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        boxes[(kThreadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        boxes[(kThreadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        boxes[(kThreadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = kThreadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = boxes + cur_box_idx * 5;
    int i = 0;
    uint64_t t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (IoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, kThreadsPerBlock);
    mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

template <typename T>
void NMS(const platform::CUDADeviceContext &ctx, const Tensor &proposals,
         const Tensor &sorted_indices, const T nms_threshold,
         Tensor *keep_out) {
  int boxes_num = proposals.dims()[0];
  PADDLE_ENFORCE_EQ(boxes_num, sorted_indices.dims()[0]);

  const int col_blocks = DIVUP(boxes_num, kThreadsPerBlock);
  dim3 blocks(DIVUP(boxes_num, kThreadsPerBlock),
              DIVUP(boxes_num, kThreadsPerBlock));
  dim3 threads(kThreadsPerBlock);

  const T *boxes = proposals.data<T>();
  Tensor d_mask;
  uint64_t *d_mask_data =
      d_mask.mutable_data<uint64_t>({boxes_num * col_blocks}, ctx.GetPlace());
  LOG(ERROR) << "========= NMS Kernel =======";
  NMSKernel<<<blocks, threads>>>(boxes_num, nms_threshold, boxes, d_mask_data);

  Tensor h_mask;
  uint64_t *h_mask_data =
      d_mask.mutable_data<uint64_t>({boxes_num * col_blocks}, ctx.GetPlace());
  TensorCopySync(d_mask, platform::CPUPlace(), &h_mask);
  LOG(ERROR) << "========= NMS Kernel end =======";

  std::vector<uint64_t> remv(col_blocks);
  memset(&remv[0], 0, sizeof(uint64_t) * col_blocks);

  LOG(ERROR) << "========= NMS remove 0 =======";

  std::vector<int> keep_vec;
  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / kThreadsPerBlock;
    int inblock = i % kThreadsPerBlock;
    // if (!(remv[nblock] & (1ULL << inblock))) {
    if (!(remv[nblock] & (1 << inblock))) {
      ++num_to_keep;
      keep_vec.push_back(i);
      uint64_t *p = &h_mask_data[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
  LOG(ERROR) << "========= NMS remove =======";

  int *keep = keep_out->mutable_data<int>({num_to_keep}, ctx.GetPlace());
  auto place = boost::get<platform::CUDAPlace>(ctx.GetPlace());
  memory::Copy(place, keep, platform::CPUPlace(), keep_vec.data(),
               sizeof(int) * num_to_keep, 0);
}

template <typename T>
std::pair<Tensor, Tensor> ProposalForOneImage(
    const platform::CUDADeviceContext &ctx, const Tensor &im_info,
    const Tensor &anchors, const Tensor &variances,
    const Tensor &bbox_deltas,  // [M, 4]
    const Tensor &scores,       // [N, 1]
    int pre_nms_top_n, int post_nms_top_n, float nms_thresh, float min_size,
    float eta) {
  // 1. pre nms
  LOG(ERROR) << "============= sort 0 =======";
  Tensor scores_sort, index_sort;
  SortDescending<T>(ctx, scores, &scores_sort, &index_sort);
  int num = scores.numel();
  int pre_nms_num = (pre_nms_top_n <= 0 || pre_nms_top_n > num) ? scores.numel()
                                                                : pre_nms_top_n;
  scores_sort.Resize({pre_nms_num, 1});
  index_sort.Resize({pre_nms_num, 1});
  // LOG(ERROR) << index_sort;

  // 2. box decode and clipping
  Tensor proposals;
  proposals.mutable_data<T>({pre_nms_num, 4}, ctx.GetPlace());
  int block = 512;
  auto stream = ctx.stream();
  BoxDecodeAndClipKernel<T><<<DIVUP(pre_nms_num, block), block, 0, stream>>>(
      anchors.data<T>(), bbox_deltas.data<T>(), variances.data<T>(),
      index_sort.data<int>(), im_info.data<T>(), pre_nms_num,
      proposals.data<T>());
  LOG(ERROR) << "============= BoxDecodeAndClipKernel ======= " << pre_nms_num;
  ctx.Wait();
  // LOG(ERROR) << proposals;

  // 3. filter
  Tensor keep_index, keep_num_t;
  keep_index.mutable_data<int>({pre_nms_num}, ctx.GetPlace());
  keep_num_t.mutable_data<int>({1}, ctx.GetPlace());
  min_size = std::max(min_size, 1.0f);
  FilterBBoxes<T, 256><<<1, 256, 0, stream>>>(
      proposals.data<T>(), im_info.data<T>(), min_size, pre_nms_num,
      keep_num_t.data<int>(), keep_index.data<int>());
  ctx.Wait();
  int keep_num;
  const auto gpu_place = boost::get<platform::CUDAPlace>(ctx.GetPlace());
  memory::Copy(platform::CPUPlace(), &keep_num, gpu_place,
               keep_num_t.data<int>(), sizeof(int), 0);
  ctx.Wait();

  LOG(ERROR) << "============= FilterBBoxes ======= " << keep_num;

  keep_index.Resize({keep_num});

  LOG(ERROR) << keep_index;

  Tensor scores_filter, proposals_filter;
  proposals_filter.mutable_data<T>({keep_num, 4}, ctx.GetPlace());
  scores_filter.mutable_data<T>({keep_num, 1}, ctx.GetPlace());
  GPUGather<T>(ctx, proposals, keep_index, &proposals_filter);
  GPUGather<T>(ctx, scores_sort, keep_index, &scores_filter);

  if (nms_thresh <= 0) {
    return std::make_pair(proposals_filter, scores_filter);
  }

  // 4. nms
  Tensor keep_nms;
  NMS<T>(ctx, proposals_filter, scores_filter, nms_thresh, &keep_nms);
  if (post_nms_top_n > 0 && post_nms_top_n < keep_nms.numel()) {
    keep_nms.Resize({post_nms_top_n});
  }
  LOG(ERROR) << "==== NMS == " << keep_nms.dims() << " " << post_nms_top_n;
  LOG(ERROR) << keep_nms;

  Tensor scores_nms, proposals_nms;
  proposals_nms.mutable_data<T>({keep_nms.numel(), 4}, ctx.GetPlace());
  scores_nms.mutable_data<T>({keep_nms.numel(), 1}, ctx.GetPlace());
  GPUGather<T>(ctx, proposals_filter, keep_nms, &proposals_nms);
  GPUGather<T>(ctx, scores_filter, keep_nms, &scores_nms);

  return std::make_pair(proposals_nms, scores_nms);
}

template <typename DeviceContext, typename T>
class CUDAGenerateProposalsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *scores = context.Input<Tensor>("Scores");
    auto *bbox_deltas = context.Input<Tensor>("BboxDeltas");
    auto *im_info = context.Input<Tensor>("ImInfo");
    auto *anchors = context.Input<Tensor>("Anchors");
    auto *variances = context.Input<Tensor>("Variances");

    auto *rpn_rois = context.Output<LoDTensor>("RpnRois");
    auto *rpn_roi_probs = context.Output<LoDTensor>("RpnRoiProbs");

    int pre_nms_top_n = context.Attr<int>("pre_nms_topN");
    int post_nms_top_n = context.Attr<int>("post_nms_topN");
    float nms_thresh = context.Attr<float>("nms_thresh");
    float min_size = context.Attr<float>("min_size");
    float eta = context.Attr<float>("eta");

    auto &dev_ctx = context.template device_context<DeviceContext>();

    auto scores_dim = scores->dims();
    int64_t num = scores_dim[0];
    int64_t c_score = scores_dim[1];
    int64_t h_score = scores_dim[2];
    int64_t w_score = scores_dim[3];

    auto bbox_dim = bbox_deltas->dims();
    int64_t c_bbox = bbox_dim[1];
    int64_t h_bbox = bbox_dim[2];
    int64_t w_bbox = bbox_dim[3];

    Tensor bbox_deltas_swap, scores_swap;
    bbox_deltas_swap.mutable_data<T>({num, h_bbox, w_bbox, c_bbox},
                                     dev_ctx.GetPlace());
    scores_swap.mutable_data<T>({num, h_score, w_score, c_score},
                                dev_ctx.GetPlace());

    math::Transpose<DeviceContext, T, 4> trans;
    std::vector<int> axis = {0, 2, 3, 1};
    trans(dev_ctx, *bbox_deltas, &bbox_deltas_swap, axis);
    trans(dev_ctx, *scores, &scores_swap, axis);

    Tensor *anchor = const_cast<framework::Tensor *>(anchors);
    anchor->Resize({anchors->numel() / 4, 4});
    Tensor *var = const_cast<framework::Tensor *>(variances);
    var->Resize({var->numel() / 4, 4});

    rpn_rois->mutable_data<T>({bbox_deltas->numel() / 4, 4},
                              context.GetPlace());
    rpn_roi_probs->mutable_data<T>({scores->numel(), 1}, context.GetPlace());

    T *rpn_rois_data = rpn_rois->data<T>();
    T *rpn_roi_probs_data = rpn_roi_probs->data<T>();

    auto place = boost::get<platform::CUDAPlace>(dev_ctx.GetPlace());

    int64_t num_proposals = 0;
    std::vector<size_t> offset(1, 0);
    for (int64_t i = 0; i < num; ++i) {
      Tensor im_info_slice = im_info->Slice(i, i + 1);
      Tensor bbox_deltas_slice = bbox_deltas_swap.Slice(i, i + 1);
      Tensor scores_slice = scores_swap.Slice(i, i + 1);

      bbox_deltas_slice.Resize({h_bbox * w_bbox * c_bbox / 4, 4});
      scores_slice.Resize({h_score * w_score * c_score, 1});

      LOG(ERROR) << "================ before ProposalForOneImage =======";
      std::pair<Tensor, Tensor> box_score_pair =
          ProposalForOneImage<T>(dev_ctx, im_info_slice, *anchor, *var,
                                 bbox_deltas_slice, scores_slice, pre_nms_top_n,
                                 post_nms_top_n, nms_thresh, min_size, eta);

      Tensor proposals = box_score_pair.first;
      Tensor scores = box_score_pair.second;

      memory::Copy(place, rpn_rois_data + num_proposals * 4, place,
                   proposals.data<T>(), sizeof(T) * num_proposals, 0);
      memory::Copy(place, rpn_roi_probs_data + num_proposals, place,
                   scores.data<T>(), sizeof(T) * num_proposals, 0);
      num_proposals += proposals.dims()[0];
      offset.emplace_back(num_proposals);
    }
    framework::LoD lod;
    lod.emplace_back(offset);
    rpn_rois->set_lod(lod);
    rpn_roi_probs->set_lod(lod);
    rpn_rois->Resize({num_proposals, 4});
    rpn_roi_probs->Resize({num_proposals, 1});
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(generate_proposals,
                        ops::CUDAGenerateProposalsKernel<
                            paddle::platform::CUDADeviceContext, float>);
