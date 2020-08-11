/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/operators/elementwise/elementwise_op_function.cu.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/elementwise/elementwise_sub_op.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

template <typename T>
struct SameDimsElemwiseSub<platform::CUDADeviceContext, T> {
  void operator()(const framework::ExecutionContext& ctx,
                  const framework::Tensor* x, const framework::Tensor* y,
                  framework::Tensor* z) {
    SubRangeFunctor<T> functor(x->data<T>(), y->data<T>(), z->data<T>());
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    platform::ForRange<platform::CUDADeviceContext> for_range(dev_ctx,
                                                              x->numel());
    for_range(functor);
  }
};

template <>
struct SameDimsElemwiseSub<platform::CUDADeviceContext, platform::float16> {
  void operator()(const framework::ExecutionContext& ctx,
                  const framework::Tensor* x, const framework::Tensor* y,
                  framework::Tensor* z) {
    auto size = x->numel();
    dim3 grid_size = dim3(((size + 1) / 2 + PADDLE_CUDA_THREAD_SIZE - 1) /
                              PADDLE_CUDA_THREAD_SIZE,
                          1);
    dim3 block_size = dim3(PADDLE_CUDA_THREAD_SIZE, 1);
    const half* x2 =
        reinterpret_cast<const half*>(x->data<platform::float16>());
    const half* y2 =
        reinterpret_cast<const half*>(y->data<platform::float16>());
    half* z2 = reinterpret_cast<half*>(z->data<platform::float16>());
    SameDimsElemwiseSubCUDAKernel<<<
        grid_size, block_size, 0,
        ctx.template device_context<platform::CUDADeviceContext>().stream()>>>(
        x2, y2, z2, size);
  }
};

template <typename T>
static __global__ void SimpleElemwiseSubGradCUDAKernel(const T* dout,
                                                       int64_t size, T* dx,
                                                       T* dy) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    dx[col] = dout[col];
    dy[col] = -dout[col];
    col += blockDim.x * gridDim.x;
  }
}

template <>
__global__ void SimpleElemwiseSubGradCUDAKernel<__half>(const __half* dout,
                                                        int64_t size,
                                                        __half* dx,
                                                        __half* dy) {
  int start = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t col = start;
  int stride = blockDim.x * gridDim.x;
  int64_t half_size = (size >> 1);
  __half2 t{0, 0};
  while (col < half_size) {
    __half2 v = reinterpret_cast<const __half2*>(dout)[col];
    reinterpret_cast<__half2*>(dx)[col] = v;
    t.x = -v.x;
    t.y = -v.y;
    reinterpret_cast<__half2*>(dy)[col] = t;
    col += stride;
  }

  if (start == 0 && (size % 2)) {
    dx[size - 1] = dout[size - 1];
    dy[size - 1] = -dout[size - 1];
  }
}

template <typename DeviceContext, typename T,
          typename std::enable_if<
              !std::is_same<T, paddle::platform::float16>::value>::type*>
typename std::enable_if<
    std::is_same<DeviceContext, plat::CUDADeviceContext>::value>::type
elementwise_sub_grad(const framework::ExecutionContext& ctx,
                     const framework::Tensor* x, const framework::Tensor* y,
                     const framework::Tensor* out,
                     const framework::Tensor* dout, framework::Tensor* dx,
                     framework::Tensor* dy) {
  dim3 block_size = dim3(PADDLE_CUDA_THREAD_SIZE, 1);
  auto size = x->numel();
  dim3 grid_size =
      dim3((size + PADDLE_CUDA_THREAD_SIZE - 1) / PADDLE_CUDA_THREAD_SIZE, 1);
  SimpleElemwiseSubGradCUDAKernel<
      T><<<grid_size, block_size, 0,
           ctx.template device_context<plat::CUDADeviceContext>().stream()>>>(
      dout->data<T>(), size, dx->mutable_data<T>(ctx.GetPlace()),
      dy->mutable_data<T>(ctx.GetPlace()));
}

template <typename DeviceContext, typename T,
          typename std::enable_if<
              std::is_same<T, paddle::platform::float16>::value>::type*>
typename std::enable_if<
    std::is_same<DeviceContext, plat::CUDADeviceContext>::value>::type
elementwise_sub_grad(const framework::ExecutionContext& ctx,
                     const framework::Tensor* x, const framework::Tensor* y,
                     const framework::Tensor* out,
                     const framework::Tensor* dout, framework::Tensor* dx,
                     framework::Tensor* dy) {
  dim3 block_size = dim3(PADDLE_CUDA_THREAD_SIZE, 1);
  auto size = x->numel();
  dim3 grid_size = dim3(((size + 1) / 2 + (PADDLE_CUDA_THREAD_SIZE)-1) /
                            (PADDLE_CUDA_THREAD_SIZE),
                        1);
  half* dx2 = reinterpret_cast<half*>(
      dx->mutable_data<platform::float16>(ctx.GetPlace()));
  half* dy2 = reinterpret_cast<half*>(
      dy->mutable_data<platform::float16>(ctx.GetPlace()));
  const half* dout2 =
      reinterpret_cast<const half*>(dout->data<platform::float16>());
  SimpleElemwiseSubGradCUDAKernel<half><<<
      grid_size, block_size, 0,
      ctx.template device_context<plat::CUDADeviceContext>().stream()>>>(
      dout2, size, dx2, dy2);
}

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    elementwise_sub,
    ops::ElementwiseSubKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseSubKernel<paddle::platform::CUDADeviceContext,
                              paddle::platform::float16>,
    ops::ElementwiseSubKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseSubKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseSubKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_sub_grad,
    ops::ElementwiseSubGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseSubGradKernel<paddle::platform::CUDADeviceContext,
                                  paddle::platform::float16>,
    ops::ElementwiseSubGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseSubGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseSubGradKernel<paddle::platform::CUDADeviceContext,
                                  int64_t>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_sub_grad_grad,
    ops::ElementwiseSubDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        float>,
    ops::ElementwiseSubDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        double>,
    ops::ElementwiseSubDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        int>,
    ops::ElementwiseSubDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                        int64_t>);
