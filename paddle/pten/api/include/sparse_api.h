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

#include "paddle/pten/api/include/tensor.h"
#include "paddle/pten/common/backend.h"

namespace paddle {
namespace experimental {
namespace sparse {

PADDLE_API Tensor to_sparse_coo(const Tensor& x,
                                Backend backend,
                                const int64_t sparse_dim);

PADDLE_API Tensor to_sparse_csr(const Tensor& x, Backend backend);

PADDLE_API Tensor to_dense(const Tensor& x, Backend backend);

}  // namespace sparse
}  // namespace experimental
}  // namespace paddle
