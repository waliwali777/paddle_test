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

#include "paddle/pten/kernels/split_kernel.h"

#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/strided_memcpy.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/pten/core/kernel_registry.h"
namespace pten {

template <typename T, typename Context>
void SplitKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const ScalarArray& num_or_sections,
                 const Scalar& axis_scalar,
                 std::vector<DenseTensor*> outs) {
  std::vector<const DenseTensor*> shape_refer;
  for (size_t j = 0; j < outs.size(); ++j) {
    outs[j]->mutable_data<T>();
    shape_refer.emplace_back(outs[j]);
  }

  int axis = axis_scalar.to<int>();
  // Sometimes direct copies will be faster, this maybe need deeply analysis.
  if (axis == 0 && outs.size() < 10) {
    paddle::operators::StridedMemcpyWithAxis0<T>(
        dev_ctx, x, shape_refer, &outs);
  } else {
    paddle::operators::math::SplitFunctor<DeviceContext, T> functor;
    functor(dev_ctx, x, shape_refer, axis, &outs);
  }
}

}  // namespace pten

PT_REGISTER_KERNEL(split,
                   CPU,
                   ALL_LAYOUT,
                   pten::SplitKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   bool,
                   paddle::platform::float16) {}
