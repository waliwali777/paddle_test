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
/*
 * copyright (C) 2022 KUNLUNXIN, Inc
 */

#pragma once
#include "xpu/xdnn.h"

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

DLL_EXPORT int add2(Context* ctx, const float* x, float* y, int len);
template <typename T>
DLL_EXPORT int fast_where(Context* ctx,
                          const bool* condition,
                          const T* x,
                          const T* y,
                          T* out,
                          int64_t len);
template <typename T, typename TID>
DLL_EXPORT int take_along_axis(Context* ctx,
                               const T* x,
                               const TID* index,
                               T* y,
                               const std::vector<int64_t>& xshape,
                               const std::vector<int64_t>& idxshape,
                               int64_t axis);

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
