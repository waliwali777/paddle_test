// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

namespace phi {
namespace fusion {
template <typename T, typename Context>
void FCKernel(const Context& dev_ctx,
              const DenseTensor& input,
              const DenseTensor& w,
              const paddle::optional<DenseTensor>& bias,
              const int in_num_col_dims,
              const std::string& activation_type,
              const bool use_mkldnn,
              const bool padding_weights,
              const bool use_quantizer,
              const std::string& mkldnn_data_type,
              const float scale_in,
              const std::vector<float>& scale_weights,
              const float scale_out,
              const bool force_fp32_output,
              DenseTensor* out);

}  // namespace fusion
}  // namespace phi
