/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {
  namespace sparse {
//  template <typename T, typename Context>
//  void TraceKernel(const Context& dev_ctx,
//                   const DenseTensor& x,
//                   int offset,
//                   int axis1,
//                   int axis2,
//                   DenseTensor* out);

  template <typename T, typename Context>
  void AddCsrKernel(const Context& dev_ctx,
                 const SparseCsrTensor& x,
                 const SparseCsrTensor& y,
                 SparseCsrTensor* out);


//  template <typename T, typename Context>
//  void AddKernel(const Context& dev_ctx,
//                 const SparseCsrTensor& x,
//                 const SparseCsrTensor& y,
//                 SparseCsrTensor* out);

//  template <typename T, typename Context>
//  SparseCooTensor MaxPool(const Context& dev_ctx,
//                          const SparseCooTensor& x,
//                          const std::vector<int>& kernel_sizes,
//                          const std::vector<int>& paddings,
//                          const std::vector<int>& dilations,
//                          const std::vector<int>& strides,
//                          DenseTensor* rulebook) {
//    DenseTensor indices = phi::Empty<Context>(
//        dev_ctx, DenseTensorMeta(DataType::INT32, {1}, DataLayout::NCHW));
//    DenseTensor values =
//        phi::Empty<Context>(dev_ctx, DenseTensorMeta(x.dtype(), {1}, x.layout()));
//    SparseCooTensor coo(indices, values, x.dims());
//    MaxPoolKernel<T, Context>(
//        dev_ctx, x, kernel_sizes, paddings, dilations, strides, &coo, rulebook);
//    return coo;
//  }

  }  // namespace sparse
}  // namespace phi
