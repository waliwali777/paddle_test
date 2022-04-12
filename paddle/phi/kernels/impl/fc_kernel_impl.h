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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/fc_functor.h"

namespace phi {

template <typename T, typename Context>
void FcKernel(const Context& dev_ctx,
              const DenseTensor& input,
              const DenseTensor& weight,
              const DenseTensor& bias,
              int in_num_col_dims,
              bool padding_weights,
              DenseTensor* out) {
  T* out_data = dev_ctx.template Alloc<T>(out);
  auto& w_dims = weight.dims();
  auto& out_dims = out->dims();
  auto w_dims0 = padding_weights ? w_dims[0] - 4 : w_dims[0];
  auto w_dims1 = padding_weights ? w_dims[1] - 4 : w_dims[1];
  int M = phi::product(out_dims) / w_dims1;
  phi::funcs::FCFunctor<Context, T> fc;

  fc(dev_ctx,
     M,
     w_dims1,
     w_dims0,
     input.data<T>(),
     weight.data<T>(),
     out_data,
     bias.data<T>(),
     false,
     padding_weights);
}

}  // namespace phi
