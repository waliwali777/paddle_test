// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/index_put_grad_kernel.h"
#include <numeric>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/array.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/reshape_kernel.h"

namespace phi {

template <typename T>
__global__ void range_cuda_kernel(int64_t N, T* out) {
  int64_t idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx >= N) {
    return;
  }
  out[idx] = idx;
}

template <typename T, typename Context>
phi::DenseTensor GetRangeCudaTensor(const Context& dev_ctx,
                                    int64_t N,
                                    phi::DataType dtype) {
  phi::DenseTensor res(dtype);
  res.Resize(phi::make_ddim({N}));
  DenseTensor* p_res = &res;
  T* out = dev_ctx.template Alloc<T>(p_res);
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, N);
  range_cuda_kernel<T>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          N, out);
  return res;
}

template <typename T, size_t Rank>
__global__ void set_zero_cuda_kernel(const int64_t N,
                                     int64_t** indices,
                                     phi::Array<int64_t, Rank> stride,
                                     phi::Array<int64_t, Rank> shape,
                                     T* out) {
  int64_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  int64_t cur_ix = 0;

  if (idx >= N) {
    return;
  }
  int64_t offset = 0;
  for (int i = 0; i < Rank; ++i) {
    cur_ix = (int64_t(*(indices[i] + idx)));
    if (cur_ix < 0) {
      cur_ix += shape[i];
    }
    offset += stride[i] * cur_ix;
  }

  *(out + offset) = 0;
}

template <typename T, size_t Rank>
__global__ void index_put_grad_cuda_kernel(const int64_t N,
                                           const T* out_grad,
                                           int64_t** indices,
                                           phi::Array<int64_t, Rank> stride,
                                           phi::Array<int64_t, Rank> shape,
                                           T* value_grad) {
  int64_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  int64_t cur_ix = 0;

  if (idx >= N) {
    return;
  }
  int64_t offset = 0;
  for (int i = 0; i < Rank; ++i) {
    cur_ix = (int64_t(*(indices[i] + idx)));
    if (cur_ix < 0) {
      cur_ix += shape[i];
    }
    offset += stride[i] * cur_ix;
  }

  *(value_grad + idx) = *(out_grad + offset);
}

template <typename T, typename Context, size_t Rank>
void LaunchIndexPutGradCudaKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& indices_v,
    const DenseTensor& out_grad,
    bool accumulate,
    DenseTensor* value_grad,
    DenseTensor* x_grad) {
  if (x_grad) {
    phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
    if (!accumulate) {
      T* x_grad_data = x_grad->data<T>();

      auto x_grad_dims = x_grad->dims();
      const int64_t numel = indices_v[0]->numel();
      auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);
      auto x_grad_stride = phi::stride(x_grad_dims);

      phi::Array<int64_t, Rank> stride_a;
      phi::Array<int64_t, Rank> shape_a;

      for (size_t idx = 0; idx < Rank; ++idx) {
        stride_a[idx] = x_grad_stride[idx];
        shape_a[idx] = x_grad_dims[idx];
      }

      auto pd_indices =
          GetDevicePointerArray<int64_t, Context>(dev_ctx, indices_v);
      set_zero_cuda_kernel<T, Rank><<<config.block_per_grid,
                                      config.thread_per_block,
                                      0,
                                      dev_ctx.stream()>>>(
          numel, pd_indices, stride_a, shape_a, x_grad_data);
    }
  }

  if (value_grad) {
    if (value_grad->numel() == 1) {
      DenseTensor tmp_value_grad(value_grad->dtype());
      tmp_value_grad.Resize(indices_v[0]->dims());

      T* tmp_value_grad_data = dev_ctx.template Alloc<T>(&tmp_value_grad);
      auto out_grad_data = out_grad.data<T>();

      auto out_grad_dims = out_grad.dims();
      const int64_t numel = indices_v[0]->numel();
      auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);
      auto out_grad_stride = phi::stride(out_grad_dims);

      phi::Array<int64_t, Rank> stride_a;
      phi::Array<int64_t, Rank> shape_a;

      for (size_t idx = 0; idx < Rank; ++idx) {
        stride_a[idx] = out_grad_stride[idx];
        shape_a[idx] = out_grad_dims[idx];
      }

      auto pd_indices =
          GetDevicePointerArray<int64_t, Context>(dev_ctx, indices_v);
      index_put_grad_cuda_kernel<T, Rank>
          <<<config.block_per_grid,
             config.thread_per_block,
             0,
             dev_ctx.stream()>>>(numel,
                                 out_grad_data,
                                 pd_indices,
                                 stride_a,
                                 shape_a,
                                 tmp_value_grad_data);

      std::vector<int> v_dims(tmp_value_grad.dims().size());
      std::iota(v_dims.begin(), v_dims.end(), 0);
      IntArray v_axis(v_dims);
      SumKernel<T, Context>(dev_ctx,
                            tmp_value_grad,
                            v_axis,
                            value_grad->dtype(),
                            false,
                            value_grad);
    } else if (value_grad->numel() == indices_v[0]->numel()) {
      T* value_grad_data = dev_ctx.template Alloc<T>(value_grad);
      auto out_grad_data = out_grad.data<T>();

      auto out_grad_dims = out_grad.dims();
      const int64_t numel = indices_v[0]->numel();
      auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);
      auto out_grad_stride = phi::stride(out_grad_dims);

      phi::Array<int64_t, Rank> stride_a;
      phi::Array<int64_t, Rank> shape_a;

      for (size_t idx = 0; idx < Rank; ++idx) {
        stride_a[idx] = out_grad_stride[idx];
        shape_a[idx] = out_grad_dims[idx];
      }

      auto pd_indices =
          GetDevicePointerArray<int64_t, Context>(dev_ctx, indices_v);
      index_put_grad_cuda_kernel<T, Rank><<<config.block_per_grid,
                                            config.thread_per_block,
                                            0,
                                            dev_ctx.stream()>>>(
          numel, out_grad_data, pd_indices, stride_a, shape_a, value_grad_data);
    } else {
      DenseTensor tmp_value_grad(value_grad->dtype());
      tmp_value_grad.Resize(indices_v[0]->dims());

      T* tmp_value_grad_data = dev_ctx.template Alloc<T>(&tmp_value_grad);
      auto out_grad_data = out_grad.data<T>();

      auto out_grad_dims = out_grad.dims();
      const int64_t numel = indices_v[0]->numel();
      auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);
      auto out_grad_stride = phi::stride(out_grad_dims);

      phi::Array<int64_t, Rank> stride_a;
      phi::Array<int64_t, Rank> shape_a;

      for (size_t idx = 0; idx < Rank; ++idx) {
        stride_a[idx] = out_grad_stride[idx];
        shape_a[idx] = out_grad_dims[idx];
      }

      auto pd_indices =
          GetDevicePointerArray<int64_t, Context>(dev_ctx, indices_v);
      index_put_grad_cuda_kernel<T, Rank>
          <<<config.block_per_grid,
             config.thread_per_block,
             0,
             dev_ctx.stream()>>>(numel,
                                 out_grad_data,
                                 pd_indices,
                                 stride_a,
                                 shape_a,
                                 tmp_value_grad_data);

      std::vector<int64_t> after_dims = phi::vectorize(tmp_value_grad.dims());
      std::vector<int64_t> before_dims = phi::vectorize(value_grad->dims());
      std::vector<int64_t> compress_dims;
      std::vector<int64_t> dims_without_1;
      int i = static_cast<int>(after_dims.size()) - 1;
      int j = static_cast<int>(before_dims.size()) - 1;
      if (i < j) {
        PADDLE_THROW(phi::errors::InvalidArgument(
            "shape of value can't not be broadcast to shape of x[indices]"));
      }

      while ((i >= 0) && (j >= 0)) {
        if (after_dims[i] == before_dims[j]) {
          dims_without_1.push_back(before_dims[j]);
          i--;
          j--;
          continue;
        } else if (before_dims[j] == 1) {
          compress_dims.push_back(i);
          i--;
          j--;
        } else {
          PADDLE_THROW(phi::errors::InvalidArgument(
              "shape of value can't not be broadcast to shape of x[indices]"));
        }
      }
      while (i >= 0) {
        compress_dims.push_back(i);
        i--;
      }

      phi::DenseTensor value_grad_dims_without1(value_grad->dtype());
      value_grad_dims_without1.Resize(phi::make_ddim(dims_without_1));
      IntArray v_axis(compress_dims);
      SumKernel<T, Context>(dev_ctx,
                            tmp_value_grad,
                            v_axis,
                            value_grad->dtype(),
                            false,
                            &value_grad_dims_without1);
      phi::ReshapeInferKernel<Context>(
          dev_ctx,
          value_grad_dims_without1,
          phi::IntArray(phi::vectorize(value_grad->dims())),
          value_grad);
    }
  }
}

template <typename T, typename Context>
void IndexPutGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const std::vector<const DenseTensor*>& indices_v,
                        const DenseTensor& value,
                        const DenseTensor& out_grad,
                        bool accumulate,
                        DenseTensor* x_grad,
                        DenseTensor* value_grad) {
  PADDLE_ENFORCE_EQ(
      x.dtype(),
      value.dtype(),
      phi::errors::InvalidArgument(
          "The data type of tensor in indices must be same to the data type "
          "of tensor x."));
  std::vector<DenseTensor> tmp_args;
  std::vector<const phi::DenseTensor*> int_indices_v =
      DealWithBoolIndices<T, Context>(dev_ctx, indices_v, &tmp_args);
  const size_t total_dims = x.dims().size();
  auto bd_dim = BroadCastTensorsDims(int_indices_v);

  std::vector<int64_t> res_dim_v(phi::vectorize(bd_dim));
  std::vector<const phi::DenseTensor*> res_indices_v(x.dims().size(), nullptr);
  std::vector<DenseTensor> tmp_res_indices_v;

  if (int_indices_v.size() < total_dims) {
    std::vector<int64_t> tmp_x_dims = phi::vectorize(x.dims());
    int len_bd_dim = bd_dim.size();
    res_dim_v.insert(res_dim_v.end(),
                     tmp_x_dims.begin() + int_indices_v.size(),
                     tmp_x_dims.end());

    std::vector<DenseTensor> reshaped_indices_v;
    for (size_t i = 0; i < int_indices_v.size(); ++i) {
      if (int_indices_v[i]->dtype() == phi::DataType::INT32) {
        reshaped_indices_v.emplace_back(phi::Cast<int, Context>(
            dev_ctx, *int_indices_v[i], phi::DataType::INT64));
      } else {
        reshaped_indices_v.emplace_back(*int_indices_v[i]);
      }
    }
    for (size_t i = len_bd_dim; i < res_dim_v.size(); ++i) {
      reshaped_indices_v.emplace_back(GetRangeCudaTensor<int64_t, Context>(
          dev_ctx, res_dim_v[i], phi::DataType::INT64));
    }
    phi::DDim res_dim = phi::make_ddim(res_dim_v);

    for (size_t i = 0; i < reshaped_indices_v.size(); ++i) {
      tmp_res_indices_v.emplace_back(
          GetReshapeAndExpandTensor<int64_t, Context>(
              dev_ctx,
              reshaped_indices_v[i],
              res_dim,
              bd_dim,
              ((i < int_indices_v.size())
                   ? 0
                   : i - int_indices_v.size() + len_bd_dim)));
    }
    for (size_t i = 0; i < res_indices_v.size(); ++i) {
      res_indices_v[i] = &tmp_res_indices_v[i];
    }
  } else {
    std::vector<DenseTensor> int_indices_v_tmp;

    for (size_t i = 0; i < int_indices_v.size(); ++i) {
      if (int_indices_v[i]->dtype() == phi::DataType::INT32) {
        int_indices_v_tmp.emplace_back(phi::Cast<int, Context>(
            dev_ctx, *int_indices_v[i], phi::DataType::INT64));
      } else {
        int_indices_v_tmp.emplace_back(*int_indices_v[i]);
      }
    }

    for (size_t i = 0; i < int_indices_v.size(); ++i) {
      if (bd_dim != int_indices_v[i]->dims()) {
        tmp_res_indices_v.emplace_back(
            DenseTensor(phi::DataType::INT64).Resize(bd_dim));
        ExpandKernel<int64_t, Context>(
            dev_ctx,
            int_indices_v_tmp[i],
            IntArray(phi::vectorize<int64_t>(bd_dim)),
            &tmp_res_indices_v[i]);
      } else {
        tmp_res_indices_v.emplace_back(int_indices_v_tmp[i]);
      }
    }

    for (size_t i = 0; i < res_indices_v.size(); ++i) {
      res_indices_v[i] = &tmp_res_indices_v[i];
    }
  }

  switch (total_dims) {
    case 1:
      LaunchIndexPutGradCudaKernel<T, Context, 1>(
          dev_ctx, res_indices_v, out_grad, accumulate, value_grad, x_grad);
      break;
    case 2:
      LaunchIndexPutGradCudaKernel<T, Context, 2>(
          dev_ctx, res_indices_v, out_grad, accumulate, value_grad, x_grad);
      break;
    case 3:
      LaunchIndexPutGradCudaKernel<T, Context, 3>(
          dev_ctx, res_indices_v, out_grad, accumulate, value_grad, x_grad);
      break;
    case 4:
      LaunchIndexPutGradCudaKernel<T, Context, 4>(
          dev_ctx, res_indices_v, out_grad, accumulate, value_grad, x_grad);
      break;
    case 5:
      LaunchIndexPutGradCudaKernel<T, Context, 5>(
          dev_ctx, res_indices_v, out_grad, accumulate, value_grad, x_grad);
      break;
    case 6:
      LaunchIndexPutGradCudaKernel<T, Context, 6>(
          dev_ctx, res_indices_v, out_grad, accumulate, value_grad, x_grad);
      break;
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "dims of input tensor should be less than 7, But received"
          "%d",
          x.dims().size()));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(index_put_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexPutGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16) {}
