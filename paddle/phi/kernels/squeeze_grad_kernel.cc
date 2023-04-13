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

#include "paddle/phi/kernels/squeeze_grad_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {
template <typename T, typename Context>
void SqueezeGradKernel(const Context& dev_ctx,
                       const DenseTensor& xshape,
                       const DenseTensor& dout,
                       const IntArray& axes,
                       DenseTensor* dx) {
  DenseTensor& xx = const_cast<DenseTensor&>(dout);
  dx->can_not_uses = xx.can_not_uses;
  if (*dx->canNotUse == false) {
    *dx->canNotUse = *xx.canNotUse;
  }
  xx.can_not_uses->insert(xx.canNotUse);

  xx.can_not_uses->insert(dx->canNotUse);

  auto xshape_dims = xshape.dims();
  auto x_dims = phi::slice_ddim(xshape_dims, 1, xshape_dims.size());

  dev_ctx.template Alloc<T>(dx);
  phi::Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
  dx->Resize(x_dims);
}
}  // namespace phi

PD_REGISTER_KERNEL(squeeze_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::SqueezeGradKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   bool,
                   int,
                   uint8_t,
                   int8_t,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(squeeze_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SqueezeGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   bool,
                   int,
                   uint8_t,
                   int8_t,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

#endif

#ifdef PADDLE_WITH_XPU
PD_REGISTER_KERNEL(squeeze_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::SqueezeGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   bool,
                   int,
                   uint8_t,
                   int8_t,
                   int64_t) {}

#endif
