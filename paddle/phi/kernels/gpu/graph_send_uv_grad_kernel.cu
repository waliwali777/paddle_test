// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/graph_send_uv_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpu/graph_send_recv_funcs.h"
#include "paddle/phi/kernels/gpu/graph_send_ue_recv_funcs.h"
#include "paddle/phi/kernels/impl/graph_send_ue_recv_kernel_impl.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

template <typename Context, typename T, typename IndexT>
void CalculateGrad(const Context& ctx,
                   const T* out_grad,
                   const IndexT* s_index,
                   const IndexT* d_index,
                   const phi::DDim& out_grad_dims,
                   const phi::DDim& x_grad_dims,
                   const std::string& compute_type,
                   int64_t index_size,
                   int64_t slice_size,
                   T* x_grad,
                   const DenseTensor& out_grad_tensor,
                   const DenseTensor& y) {
  std::vector<int64_t> reduce_idx;
  bool reduce = ReduceGrad(out_grad_dims, x_grad_dims, reduce_idx);

  if (compute_type == "ADD") {
#ifdef PADDLE_WITH_HIP
    int block = 256;
#else
    int block = 1024;
#endif
    int64_t n = slice_size * index_size;
    int max_grid_dimx = ctx.GetCUDAMaxGridDimSize()[0];
    int64_t grid_tmp = (n + block - 1) / block;
    int64_t grid = grid_tmp < max_grid_dimx ? grid_tmp : max_grid_dimx;
    GraphSendRecvSumCUDAFunctor<T, IndexT> functor;
    if (!reduce) {
      GraphSendRecvCUDAKernel<T, IndexT, GraphSendRecvSumCUDAFunctor<T, IndexT>>
          <<<grid, block, 0, ctx.stream()>>>(out_grad,
                                             d_index,
                                             s_index,
                                             x_grad,
                                             index_size,
                                             slice_size,
                                             functor);
    } else {
      const auto& bcast_info = phi::CalcBCastInfo(out_grad_dims, x_grad_dims);
      DenseTensor x_grad_v2 = phi::EmptyLike<T, Context>(ctx, out_grad_tensor);
      phi::funcs::SetConstant<Context, T>()(ctx, &x_grad_v2, T(0));
      T* x_grad_v2_data = x_grad_v2.data<T>();
      GraphSendRecvCUDAKernel<T, IndexT, GraphSendRecvSumCUDAFunctor<T, IndexT>>
          <<<grid, block, 0, ctx.stream()>>>(out_grad,
                                             d_index,
                                             s_index,
                                             x_grad,
                                             index_size,
                                             bcast_info.out_len,
                                             functor);
      // Run reduce sum
      DenseTensor x_grad_out = phi::Sum<T, Context>(
          ctx,
          x_grad_v2,
          reduce_idx,
          paddle::experimental::CppTypeToDataType<T>::Type(),
          true);
#ifdef PADDLE_WITH_HIP
      hipMemcpy(x_grad,
                x_grad_out.data<T>(),
                x_grad_out.numel() * sizeof(T),
                hipMemcpyDeviceToDevice);
#else
      cudaMemcpy(x_grad,
                 x_grad_out.data<T>(),
                 x_grad_out.numel() * sizeof(T),
                 cudaMemcpyDeviceToDevice);
#endif
    }
  } else if (compute_type == "MUL") {
    const auto& bcast_info = phi::CalcBCastInfo(y.dims(), out_grad_dims);
    thrust::device_vector<int64_t> l_bcastoff, r_bcastoff;
    if (bcast_info.use_bcast) {
      CopyBCastOff(bcast_info, l_bcastoff, r_bcastoff);
    }
    int64_t out_len = bcast_info.out_len;
    const int ntx = FindNumThreads(out_len);
    const int nty = CUDA_MAX_NUM_THREADS / ntx;
    const int nbx = (out_len + ntx - 1) / ntx;
    const int nby = (index_size + nty - 1) / nty;
    const dim3 grid_(nbx, nby);
    const dim3 block_(ntx, nty);
    funcs::MultiplyFunctor<T> mul_functor;
    GraphSendUERecvSumCUDAFunctor<T> sum_functor;
    const T* y_data = y.data<T>();
    if (!reduce) {
      GraphSendUERecvCUDAKernel<T,
                                IndexT,
                                GraphSendUERecvSumCUDAFunctor<T>,
                                funcs::MultiplyFunctor<T>>
          <<<grid_, block_, 0, ctx.stream()>>>(
              y_data,
              out_grad,
              d_index,
              s_index,
              thrust::raw_pointer_cast(l_bcastoff.data()),
              thrust::raw_pointer_cast(r_bcastoff.data()),
              x_grad,
              index_size,
              bcast_info.l_len,
              bcast_info.r_len,
              out_len,
              bcast_info.use_bcast,
              mul_functor,
              sum_functor);
    } else {
      DenseTensor x_grad_v2 = phi::EmptyLike<T, Context>(ctx, out_grad_tensor);
      phi::funcs::SetConstant<Context, T>()(ctx, &x_grad_v2, T(0));
      T* x_grad_v2_data = x_grad_v2.data<T>();
      GraphSendUERecvCUDAKernel<T,
                                IndexT,
                                GraphSendUERecvSumCUDAFunctor<T>,
                                funcs::MultiplyFunctor<T>>
          <<<grid_, block_, 0, ctx.stream()>>>(
              y_data,
              out_grad,
              d_index,
              s_index,
              thrust::raw_pointer_cast(l_bcastoff.data()),
              thrust::raw_pointer_cast(r_bcastoff.data()),
              x_grad_v2_data,
              index_size,
              bcast_info.l_len,
              bcast_info.r_len,
              out_len,
              bcast_info.use_bcast,
              mul_functor,
              sum_functor);
      // Run reduce_sum
      DenseTensor x_grad_out = phi::Sum<T, Context>(
          ctx,
          x_grad_v2,
          reduce_idx,
          paddle::experimental::CppTypeToDataType<T>::Type(),
          true);
#ifdef PADDLE_WITH_HIP
      hipMemcpy(x_grad,
                x_grad_out.data<T>(),
                x_grad_out.numel() * sizeof(T),
                hipMemcpyDeviceToDevice);
#else
      cudaMemcpy(x_grad,
                 x_grad_out.data<T>(),
                 x_grad_out.numel() * sizeof(T),
                 cudaMemcpyDeviceToDevice);
#endif
    }
  }
}

template <typename Context, typename T, typename IndexT>
void GraphSendUVGradOpCUDAKernelLaunchHelper(const Context& ctx,
                                             const DenseTensor& x,
                                             const DenseTensor& y,
                                             const DenseTensor& out_grad,
                                             const DenseTensor& src_index,
                                             const DenseTensor& dst_index,
                                             const std::string& compute_type,
                                             DenseTensor* x_grad,
                                             DenseTensor* y_grad) {
  const int& index_size = dst_index.dims()[0];

  ctx.template Alloc<T>(x_grad);
  T* x_grad_data = x_grad->data<T>();
  ctx.template Alloc<T>(y_grad);
  T* y_grad_data = y_grad->data<T>();
  const auto& x_grad_dims = x_grad->dims();
  const auto& y_grad_dims = y_grad->dims();
  int64_t memset_size_x = 1, memset_size_y = 1;
  int64_t slice_size_x = 1, slice_size_y = 1;
  for (int i = 0; i < x_grad_dims.size(); i++) {
    memset_size_x *= x_grad_dims[i];
    if (i > 0) slice_size_x *= x_grad_dims[i];
  }
  for (int i = 0; i < y_grad_dims.size(); i++) {
    memset_size_y *= y_grad_dims[i];
    if (i > 0) slice_size_y *= y_grad_dims[i];
  }
  const size_t& memset_bytes_x = memset_size_x * sizeof(T);
  const size_t& memset_bytes_y = memset_size_y * sizeof(T);
#ifdef PADDLE_WITH_HIP
  hipMemset(x_grad_data, 0, memset_bytes_x);
  hipMemset(y_grad_data, 0, memset_bytes_y);
#else
  cudaMemset(x_grad_data, 0, memset_bytes_x);
  cudaMemset(y_grad_data, 0, memset_bytes_y);
#endif

  if (index_size == 0) return;

  const T* out_grad_data = out_grad.data<T>();
  const IndexT* s_index = src_index.data<IndexT>();
  const IndexT* d_index = dst_index.data<IndexT>();
  // Calculate X grad.
  const auto& out_grad_dims = out_grad.dims();
  CalculateGrad<Context, T, IndexT>(ctx,
                                    out_grad_data,
                                    s_index,
                                    d_index,
                                    out_grad_dims,
                                    x_grad_dims,
                                    compute_type,
                                    index_size,
                                    slice_size_x,
                                    x_grad_data,
                                    out_grad,
                                    y);
  // Calculate Y grad.
  CalculateGrad<Context, T, IndexT>(ctx,
                                    out_grad_data,
                                    d_index,
                                    s_index,
                                    out_grad_dims,
                                    y_grad_dims,
                                    compute_type,
                                    index_size,
                                    slice_size_y,
                                    y_grad_data,
                                    out_grad,
                                    x);
}

template <typename T, typename Context>
void GraphSendUVGradKernel(const Context& ctx,
                           const DenseTensor& x,
                           const DenseTensor& y,
                           const DenseTensor& src_index,
                           const DenseTensor& dst_index,
                           const DenseTensor& out_grad,
                           const std::string& compute_type,
                           DenseTensor* x_grad,
                           DenseTensor* y_grad) {
  auto index_type = src_index.dtype();
  if (index_type == phi::DataType::INT32) {
    GraphSendUVGradOpCUDAKernelLaunchHelper<Context, T, int32_t>(ctx,
                                                                 x,
                                                                 y,
                                                                 out_grad,
                                                                 src_index,
                                                                 dst_index,
                                                                 compute_type,
                                                                 x_grad,
                                                                 y_grad);
  } else if (index_type == phi::DataType::INT64) {
    GraphSendUVGradOpCUDAKernelLaunchHelper<Context, T, int64_t>(ctx,
                                                                 x,
                                                                 y,
                                                                 out_grad,
                                                                 src_index,
                                                                 dst_index,
                                                                 compute_type,
                                                                 x_grad,
                                                                 y_grad);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(graph_send_uv_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::GraphSendUVGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
