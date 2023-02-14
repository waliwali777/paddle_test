
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

#pragma once

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/infermeta/unary.h"

namespace phi {

template <typename T, typename Context>
void SqueezeInferKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const IntArray& axes,
                        DenseTensor* out);

template <typename T, typename Context>
void SqueezeKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const IntArray& axes,
                   DenseTensor* out,
                   DenseTensor* xshape);

template <typename T, typename Context>
void Squeeze(const Context& dev_ctx,
             const DenseTensor& x,
             const IntArray& axes,
             DenseTensor* out) {
  MetaTensor meta_out(out);
  SqueezeInferMeta(x, axes, &meta_out);
  SqueezeInferKernel<T, Context>(dev_ctx, x, axes, out);
  DenseTensor& xx = const_cast<DenseTensor&>(x);
  out->inplace_version_counter_ = xx.inplace_version_counter_;

  xx.can_not_uses.push_back(out->can_not_use_);
  out->can_not_uses.push_back(xx.can_not_use_);
}

}  // namespace phi
