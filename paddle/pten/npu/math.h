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

#pragma once

#ifdef PADDLE_WITH_ASCEND_CL

#include "paddle/pten/core/dense_tensor.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/platform/device_context.h"

namespace pt {

using NPUDeviceContext = paddle::platfrom::NPUDeviceContext;

template <typename T>
void Mean(const NPUDeviceContext& dev_ctx,
          const DenseTensor& x,
          DenseTensor* out) {
  std::vector<int> axes;
  framework::NPUAttributeMap attr_input = {{"keep_dims", false},
                                           {"axes", axes}};
  out->mutable_data<T>();
  const auto& runner = NpuOpRunner("ReduceMeanD", {x}, {*out}, attr_input);
  auto stream =
      ctx.template device_context<paddle::platform::NPUDeviceContext>()
          .stream();
  runner.Run(stream);
}

template <typename T>
void Scale(const NPUDeviceContext& dev_ctx,
           const DenseTensor& x,
           float scale,
           float bias,
           bool bias_after_scale,
           DenseTensor* out) {
  out->mutable_data<T>();
  auto stream = dev_ctx.stream();
  float _power = 1.0;
  if (bias_after_scale) {
    auto runner =
        NpuOpRunner("Power",
                    {x},
                    {*out},
                    {{"power", _power}, {"scale", scale}, {"shift", bias}});

    runner.Run(stream);
  } else {
    DenseTensor tmp_x(std::unique_ptr<TensorMeta>(
        new TensorMeta(x.dims(), x.backend(), x.type(), x.layout())));
    tmp_x.mutable_data<T>();

    auto runner_tmp = NpuOpRunner("Adds", {x}, {tmp_x}, {{"value", bias}});
    runner_tmp.Run(stream);

    out->mutable_data<T>(x.place());
    float _bias = 0.0;
    auto runner =
        NpuOpRunner("Power",
                    {tmp_x},
                    {*out},
                    {{"power", _power}, {"scale", scale}, {"shift", _bias}});
    runner.Run(stream);
  }
}

}  // namespace pt

#endif
