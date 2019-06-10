// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>
#include <vector>
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/core/scope.h"
#include "paddle/fluid/lite/operators/op_params.h"
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace lite {
namespace operators {

class PoolOpLite : public OpLite {
 public:
  PoolOpLite() {}

  explicit PoolOpLite(const std::string &type) : OpLite(type) {}

  bool CheckShape() const override;

  bool InferShape() const override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }
  // TODO(Superjomn) replace framework::OpDesc with a lite one.
  bool AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) override {
    auto input = op_desc.Input("X").front();
    auto out = op_desc.Output("Out").front();

    param_.x = scope->FindVar(input)->GetMutable<Tensor>();
    param_.output = scope->FindVar(out)->GetMutable<Tensor>();
    param_.pooling_type = op_desc.GetAttr<std::string>("pooling_type");
    param_.ksize = op_desc.GetAttr<std::vector<int>>("ksize");
    param_.strides = op_desc.GetAttr<std::vector<int>>("strides");
    param_.paddings = op_desc.GetAttr<std::vector<int>>("paddings");
    param_.ceil_mode = op_desc.GetAttr<bool>("ceil_mode");
    param_.adaptive = op_desc.GetAttr<bool>("adaptive");
    param_.global_pooling = op_desc.GetAttr<bool>("global_pooling");
    return true;
  }

  std::string DebugString() const override { return "pool"; }

 private:
  mutable PoolParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
