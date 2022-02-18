/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/api/include/tensor.h"

namespace paddle {
namespace experimental {

// declare cast api
Tensor cast(const Tensor &x, DataType out_dtype);

Tensor Tensor::cast(DataType target_type) const {
  return experimental::cast(*this, target_type);
}

}  // namespace experimental
}  // namespace paddle
