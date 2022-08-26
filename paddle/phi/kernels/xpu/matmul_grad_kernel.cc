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

#include "paddle/phi/kernels/matmul_grad_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace phi {

template <typename T, typename Context>
void MatmulGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      const DenseTensor& dout,
                      bool transpose_x,
                      bool transpose_y,
                      DenseTensor* dx,
                      DenseTensor* dy) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  if (dx) {
    dev_ctx.template Alloc<T>(dx);
  }
  if (dy) {
    dev_ctx.template Alloc<T>(dy);
  }

  const XPUType* dout_ptr = reinterpret_cast<const XPUType*>(dout.data<T>());
  const XPUType* x_ptr = reinterpret_cast<const XPUType*>(x.data<T>());
  const XPUType* y_ptr = reinterpret_cast<const XPUType*>(y.data<T>());

  xpu::Context* xpu_ctx = dev_ctx.x_context();

  XpuFcInfo info_forward;
  GetFCInfo(x.dims(), y.dims(), transpose_x, transpose_y, &info_forward);
  xpu::ctx_guard RAII_GUARD(xpu_ctx);
  // begin calculate
  const XPUType* a_1 = reinterpret_cast<const XPUType*>(NULL);
  const XPUType* b_1 = reinterpret_cast<const XPUType*>(NULL);
  const XPUType* a_2 = reinterpret_cast<const XPUType*>(NULL);
  const XPUType* b_2 = reinterpret_cast<const XPUType*>(NULL);
  XPUType* c_1 = (dx == NULL) ? reinterpret_cast<XPUType*>(NULL)
                              : reinterpret_cast<XPUType*>(dx->data<T>());
  XPUType* c_2 = (dy == NULL) ? reinterpret_cast<XPUType*>(NULL)
                              : reinterpret_cast<XPUType*>(dy->data<T>());
  XpuFcInfo info_dx;
  XpuFcInfo info_dy;
  std::tuple<XpuFcInfo,
             XpuFcInfo,
             const XPUType*,
             const XPUType*,
             const XPUType*,
             const XPUType*>
      fc_info = MatmulGradFcInfo(xpu_ctx,
                                 &RAII_GUARD,
                                 info_forward,
                                 transpose_x,
                                 transpose_y,
                                 x_ptr,
                                 y_ptr,
                                 dout_ptr);
  std::tie(info_dx, info_dy, a_1, b_1, a_2, b_2) = fc_info;
  if (dx) {
    MatMulXPUFunction<XPUType>(xpu_ctx, a_1, b_1, c_1, info_dx, 1.0f);
  }
  if (dy) {
    MatMulXPUFunction<XPUType>(xpu_ctx, a_2, b_2, c_2, info_dy, 1.0f);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    matmul_grad, XPU, ALL_LAYOUT, phi::MatmulGradKernel, float, phi::float16) {}
