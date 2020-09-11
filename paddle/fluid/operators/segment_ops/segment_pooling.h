/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include <string>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T, typename IndexT>
class SegmentPoolFunctor {
 public:
  /* max in pool has index output */
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  const framework::Tensor& segments, framework::Tensor* output,
                  framework::Tensor* index = nullptr,
                  const std::string pooltype = "SUM");
};

template <typename DeviceContext, typename T, typename IndexT>
class SegmentPoolGradFunctor {
 public:
  /* max min pool has index*/
  void operator()(const DeviceContext& context,
                  const framework::Tensor& out_grad,
                  const framework::Tensor& segments, framework::Tensor* in_grad,
                  const framework::Tensor* index = nullptr,
                  const std::string pooltype = "SUM");
};

}  // namespace operators
}  // namespace paddle
