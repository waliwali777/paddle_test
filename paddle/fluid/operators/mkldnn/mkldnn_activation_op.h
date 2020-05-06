/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename Functor>
class MKLDNNActivationKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_NOT_NULL(
        context.Input<framework::Tensor>("X"),
        platform::errors::NotFound(
            "Cannot find input tensor X, variable name = %s not found.",
            context.InputName("X")));
    PADDLE_ENFORCE_NOT_NULL(
        context.Output<framework::Tensor>("Out"),
        platform::errors::NotFound(
            "Cannot find output tensor Out, variable name = %s not found.",
            context.OutputName("Out")));
    Functor functor;

    auto attrs = functor.GetAttrs();
    for (auto& attr : attrs) {
      *attr.second = context.Attr<float>(attr.first);
    }
    functor(context);
  }
};

template <typename Functor>
class MKLDNNActivationGradKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    Functor functor;

    auto attrs = functor.GetAttrs();
    for (auto& attr : attrs) {
      *attr.second = context.Attr<float>(attr.first);
    }
    functor(context);
  }
};

}  // namespace operators
}  // namespace paddle
