/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cusparse.h>
#include <iostream>
#include <limits>
#include <vector>
#include "paddle/fluid/operators/sparse_attention_op.h"
#include "paddle/fluid/platform/dynload/cusparse.h"

namespace ops = paddle::operators;
namespace plf = paddle::platform;

namespace paddle {
namespace operators {

template <typename T>
__forceinline__ __device__ T CudaShuffleXorSync(unsigned mask, T val,
                                                int width = warpSize) {
#if defined(PADDLE_WITH_HIP) || CUDA_VERSION < 9000
  return __shfl_xor(val, width);
#else
  return __shfl_xor_sync(mask, val, width);
#endif
}

template <typename T, int BatchSize, int WarpSize>
__device__ __forceinline__ void WarpReduceSum(T* sum) {
#pragma unroll
  for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < BatchSize; ++i) {
      T sum_val = CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = sum[i] + sum_val;
    }
  }
}

template <typename T, int BatchSize, int WarpSize>
__device__ __forceinline__ void WarpReduceMax(T* sum) {
#pragma unroll
  for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < BatchSize; ++i) {
      T max_val = CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = max(sum[i], max_val);
    }
  }
}

template <typename T, int BlockSize, int NnzBlockMax, bool KpMode = false,
          bool AttnMode = false>
__global__ void BlockSparseSoftmaxForward(T* softmax, const T* src, T scale,
                                          const T* kp_mask, const T* attn_mask,
                                          const int* layout_rowptr,
                                          const int* layout_colindex,
                                          int seq_len) {
  // constants
  const int WarpSize = 32;
  const int BlockSize2 = BlockSize * BlockSize;

  // current thread related info
  const int cur = blockIdx.x * blockDim.y + threadIdx.y;
  const int cur_batch = cur / seq_len;
  const int cur_seq = cur % seq_len;
  const int cur_seqb = cur / BlockSize;
  const int cur_inb = cur % BlockSize;

  // number of nnz block in cur_seqb
  const int cur_nnzb = layout_rowptr[cur_seqb + 1] - layout_rowptr[cur_seqb];

  T srcdata[(BlockSize * NnzBlockMax + WarpSize - 1) / WarpSize];
  T attndata[(BlockSize * NnzBlockMax + WarpSize - 1) / WarpSize];

  // read kp mask
  T datakp_mask = (KpMode == true) ? kp_mask[cur] : 0;

  // read tensor data, attn mask
  if (BlockSize == 1) {  // BlockSize = 1
    const int Iter = (cur_nnzb + WarpSize - 1) / WarpSize;
    const T* srcptr = src + layout_rowptr[cur_seqb];
    const T* attnptr = attn_mask + cur_seqb * seq_len;
    const int* colindex = layout_colindex + layout_rowptr[cur_seqb];
    for (int j = 0; j < Iter; j++) {
      int xidx = j * WarpSize + threadIdx.x;
      int didx = j;
      if (xidx < cur_nnzb) {
        if (AttnMode == true) {
          if (std::abs(attnptr[colindex[xidx]]) <
              std::numeric_limits<T>::epsilon()) {
            srcdata[didx] =
                -std::numeric_limits<T>::infinity() * scale + datakp_mask;
          } else {
            srcdata[didx] = scale * srcptr[xidx] + datakp_mask;
          }
        } else {
          srcdata[didx] = scale * srcptr[xidx] + datakp_mask;
        }
      } else {
        srcdata[didx] = -std::numeric_limits<T>::infinity();
      }
    }
  } else if (BlockSize == 64 || BlockSize == 32) {  // BlockSize = 64, 32
    for (int i = 0; i < cur_nnzb; i++) {
      const T* srcptr = src + layout_rowptr[cur_seqb] * BlockSize2 +
                        i * BlockSize2 + cur_inb * BlockSize;
      const T* attnptr =
          attn_mask + cur_seq * seq_len + layout_colindex[cur_seqb] * BlockSize;
      const int IterPerBlock = BlockSize / WarpSize;
#pragma unroll
      for (int j = 0; j < IterPerBlock; j++) {
        int didx = i * IterPerBlock + j;
        int xidx = threadIdx.x + j * WarpSize;

        if (AttnMode == true) {
          if (std::abs(attnptr[xidx]) < std::numeric_limits<T>::epsilon()) {
            srcdata[didx] =
                -std::numeric_limits<T>::infinity() * scale + datakp_mask;
          } else {
            srcdata[didx] = scale * srcptr[xidx] + datakp_mask;
          }
        } else {
          srcdata[didx] = scale * srcptr[xidx] + datakp_mask;
        }
      }
    }
  }

  // max value
  T max_value = srcdata[0];
  const int kIteration = (cur_nnzb * BlockSize + WarpSize - 1) / WarpSize;
#pragma unroll
  for (int it = 1; it < kIteration; ++it) {
    max_value = (max_value > srcdata[it]) ? max_value : srcdata[it];
  }
  WarpReduceMax<T, 1, WarpSize>(&max_value);

  // exp sum
  T sum = 0;
#pragma unroll
  for (int it = 0; it < kIteration; ++it) {
    srcdata[it] = std::exp(srcdata[it] - max_value);
    sum += srcdata[it];
  }
  WarpReduceSum<T, 1, WarpSize>(&sum);

  // compute softmax and write out
  if (BlockSize == 1) {  // BlockSize = 1
    const int Iter = (cur_nnzb + WarpSize - 1) / WarpSize;
    T* softmaxptr = softmax + layout_rowptr[cur_seqb];
    for (int j = 0; j < Iter; j++) {
      int xidx = j * WarpSize + threadIdx.x;
      int didx = j;
      if (xidx < cur_nnzb) {
        softmaxptr[xidx] = srcdata[didx] / sum;
      }
    }
  } else if (BlockSize == 64 || BlockSize == 32) {  // BlockSize = 64, 32
    for (int i = 0; i < cur_nnzb; i++) {
      T* softmaxptr = softmax + layout_rowptr[cur_seqb] * BlockSize2 +
                      i * BlockSize2 + cur_inb * BlockSize;
      const int IterPerBlock = BlockSize / WarpSize;
      for (int j = 0; j < IterPerBlock; j++) {
        int didx = i * IterPerBlock + j;
        int xidx = threadIdx.x + j * WarpSize;
        softmaxptr[xidx] = srcdata[didx] / sum;
      }
    }
  }
}

using Tensor = framework::Tensor;

#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11020
/*
input: sparse C in CSR format (num_rows,num_rows)
output: sparse C after softmax operation
*/
template <typename DeviceContext, typename T>
void sparse_softmax(const platform::CUDADeviceContext& ctx,
                    const Tensor* offset, const Tensor* columns, Tensor* input,
                    Tensor* output, const int blocksize, const int num_rows) {
  const int* offset_data = offset->data<int>();
  const int* columns_data = columns->data<int>();
  T* input_data = input->data<T>();
  T* output_data = output->data<T>();

  const int BlockSize = 1;
  dim3 blocks(32, 4, 1);
  int grid = num_rows * BlockSize / 4;
  BlockSparseSoftmaxForward<float, BlockSize, 4><<<grid, blocks>>>(
      output_data, input_data, 1.0, NULL, NULL, offset_data, columns_data,
      num_rows);
}

/*
input: dense A (num_rows,num_cols), dense B (num_rows,num_cols)
output: sparse C in CSR format (num_rows,num_rows)
*/
template <typename DeviceContext, typename T>
void Dot_sdd(const platform::CUDADeviceContext& ctx, const Tensor* A,
             const Tensor* B, const Tensor* C_offset, const Tensor* C_columns,
             Tensor* C_value, const int num_rows, const int num_cols) {
  const T* A_data = A->data<T>();
  const T* B_data = B->data<T>();

  const int* C_offset_data = C_offset->data<int>();
  const int* C_columns_data = C_columns->data<int>();

  T* C_value_data = C_value->data<T>();

  cusparseHandle_t handle = NULL;
  cusparseDnMatDescr_t matA, matB;
  cusparseSpMatDescr_t matC;

  void* dBuffer = NULL;
  size_t bufferSize = 0;

  platform::dynload::cusparseCreate(&handle);
  // Create dense matrix A
  //  the data_type must be float
  platform::dynload::cusparseCreateDnMat(&matA, num_rows, num_cols, num_cols,
                                         const_cast<T*>(A_data), CUDA_R_32F,
                                         CUSPARSE_ORDER_ROW);
  // Create dense matrix B

  platform::dynload::cusparseCreateDnMat(&matB, num_cols, num_rows, num_rows,
                                         const_cast<T*>(B_data), CUDA_R_32F,
                                         CUSPARSE_ORDER_ROW);
  // Create sparse matrix C in CSR format
  int C_nnz = C_columns->dims()[0];
  platform::dynload::cusparseCreateCsr(
      &matC, num_rows, num_rows, C_nnz, const_cast<int*>(C_offset_data),
      const_cast<int*>(C_columns_data), C_value_data, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  // allocate an external buffer if needed
  float alpha = 1.0f;
  float beta = 0.0f;

  platform::dynload::cusparseSDDMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize);
  cudaMalloc(&dBuffer, bufferSize);

  platform::dynload::cusparseSDDMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                   matA, matB, &beta, matC, CUDA_R_32F,
                                   CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer);

  platform::dynload::cusparseDestroyDnMat(matA);
  platform::dynload::cusparseDestroyDnMat(matB);
  platform::dynload::cusparseDestroySpMat(matC);
  platform::dynload::cusparseDestroy(handle);
  cudaFree(dBuffer);
}

/*
input: sparse A in CSR format (num_rows,num_rows), dense B (num_rows,num_cols)
output: dense C (num_rows,num_cols)
*/
template <typename DeviceContext, typename T>
void Dot_dsd(const platform::CUDADeviceContext& ctx, const Tensor* A_offset,
             const Tensor* A_columns, Tensor* A_value, const Tensor* B,
             Tensor* C, const int num_rows, const int num_cols) {
  const int* A_offset_data = A_offset->data<int>();
  const int* A_columns_data = A_columns->data<int>();
  T* A_value_data = A_value->data<T>();

  const T* B_data = B->data<T>();
  T* C_data = C->data<T>();

  cusparseHandle_t handle = NULL;
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;

  void* dBuffer = NULL;
  size_t bufferSize = 0;

  platform::dynload::cusparseCreate(&handle);

  // Create sparse matrix A in CSR format
  int A_nnz = A_columns->dims()[0];
  platform::dynload::cusparseCreateCsr(
      &matA, num_rows, num_rows, A_nnz, const_cast<int*>(A_offset_data),
      const_cast<int*>(A_columns_data), A_value_data, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  // Create dense matrix B
  platform::dynload::cusparseCreateDnMat(&matB, num_rows, num_cols, num_cols,
                                         const_cast<T*>(B_data), CUDA_R_32F,
                                         CUSPARSE_ORDER_ROW);
  // Create dense matrix C
  platform::dynload::cusparseCreateDnMat(&matC, num_rows, num_cols, num_cols,
                                         C_data, CUDA_R_32F,
                                         CUSPARSE_ORDER_ROW);

  // allocate an external buffer if needed
  float alpha = 1.0f;
  float beta = 0.0f;

  // allocate an external buffer if needed
  platform::dynload::cusparseSpMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
  cudaMalloc(&dBuffer, bufferSize);

  platform::dynload::cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                  matA, matB, &beta, matC, CUDA_R_32F,
                                  CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);

  platform::dynload::cusparseDestroySpMat(matA);
  platform::dynload::cusparseDestroyDnMat(matB);
  platform::dynload::cusparseDestroyDnMat(matC);
  platform::dynload::cusparseDestroy(handle);
  cudaFree(dBuffer);
}
#endif

template <typename DeviceContext, typename T>
class SparseAttentionCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11020
    auto ins = ctx.MultiInput<framework::Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    auto query = ins[0];
    auto key = ins[1];
    auto value = ins[2];
    auto offset = ins[3];
    auto columns = ins[4];

    auto query_dims = query->dims();
    int batch_size = query_dims[0];
    int num_heads = query_dims[1];
    int N = query_dims[2];
    int M = query_dims[3];
    int stride = N * M;

    T* out_data = output->mutable_data<T>(ctx.GetPlace());

    const int iter_num = batch_size * num_heads;
    for (int i = 0; i < iter_num; i++) {
      const int now_stride = i * stride;
      int nnz_num = columns->dims()[0];

      Tensor A_value;
      const std::vector<int> A_value_dims = {nnz_num};
      auto A_value_dim = framework::make_ddim(A_value_dims);
      A_value.Resize(A_value_dim);
      A_value.mutable_data<T>(ctx.GetPlace());

      const auto& dev_ctx = ctx.cuda_device_context();

      // step1:sdd
      Dot_sdd<DeviceContext, T>(dev_ctx, query, key, offset, columns, &A_value,
                                N, M);

      // step2:softmax
      Tensor B_value;
      const std::vector<int> B_value_dims = {nnz_num};
      auto B_value_dim = framework::make_ddim(B_value_dims);
      B_value.Resize(B_value_dim);
      B_value.mutable_data<T>(ctx.GetPlace());

      sparse_softmax<DeviceContext, T>(dev_ctx, offset, columns, &A_value,
                                       &B_value, 1, N);

      // step3:dsd
      Dot_dsd<DeviceContext, T>(dev_ctx, offset, columns, &B_value, value,
                                output, N, M);
    }
#endif
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    sparse_attention,
    ops::SparseAttentionCUDAKernel<plf::CUDADeviceContext, float>);
