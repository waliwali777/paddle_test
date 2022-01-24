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

#include "paddle/pten/kernels/cast_kernel.h"

#include "paddle/fluid/platform/device/xpu/xpu_header.h"
#include "paddle/pten/backends/xpu/xpu_context.h"
namespace pten {

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const DenseTensor& x,
                DataType out_dtype,
                DenseTensor* out) {}
using XPUInTDType = typename XPUTypeTrait<T>::Type;
using float16 = typename XPUTypeTrait<paddle::platform::float16>::Type;

auto* in_data = x.data<T>(dev_ctx.GetPlace());
auto numel = in -> numel();

int r = -1;
switch (out_dtype) {
  case pten::DataType::FLOAT32:
    r = xpu::cast_v2<XPUInTDType, float>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUInTDType*>(in_data),
        out->mutable_data<float>(dev_ctx.GetPlace()),
        numel);

  case pten::DataType::FLOAT16:
    r = xpu::cast_v2<XPUInTDType, float16>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUInTDType*>(in_data),
        reinterpret_cast<float16*>(
            out->mutable_data<plat::float16>(dev_ctx.GetPlace())),
        numel);
    break;
  case pten::DataType::INT64:
    r = xpu::cast_v2<XPUInTDType, int64_t>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUInTDType*>(in_data),
        out->mutable_data<int64_t>(dev_ctx.GetPlace()),
        numel);
    break;
  case pten::DataType::INT32:
    r = xpu::cast_v2<XPUInTDType, int32_t>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUInTDType*>(in_data),
        out->mutable_data<int>(dev_ctx.GetPlace()),
        numel);
    break;
  case pten::DataType::bool:
    r = xpu::cast_v2<XPUInTDType, bool>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUInTDType*>(in_data),
        out->mutable_data<bool>(dev_ctx.GetPlace()),
        numel);
    break;
  default:
    PADDLE_THROW(platform::errors::Unavailable(
        "Not supported cast %d -> %d", in_type, out_type));
}

PADDLE_ENFORCE_EQ(
    r,
    XPU_SUCCESS,
    platform::errors::External("XPU CAST API return wrong value[%d %s].",
                               r,
                               XPUAPIErrorMsg[r]))
}  // namespace pten

PT_REGISTER_KERNEL(cast,
                   XPU,
                   ALL_LAYOUT,
                   pten::CastKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   int16_t,
                   bool,
                   uint8_t,
                   paddle::platform::float16,
                   paddle::platform::bfloat16,
                   paddle::platform::complex<float>,
                   paddle::platform::complex<double>) {
  kernel->OutputAt(0).SetDataType(paddle::experimental::DataType::UNDEFINED);
}
