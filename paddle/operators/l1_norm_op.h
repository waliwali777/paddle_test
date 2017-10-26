/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

// Out = sum(abs(X))
template <typename Place, typename T>
class L1NormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const framework::Tensor *X = context.Input<framework::Tensor>("X");
    framework::Tensor *Out = context.Output<framework::Tensor>("Out");
    Out->mutable_data<T>(context.GetPlace());

    auto x = framework::EigenVector<T>::Flatten(*X);
    auto out = framework::EigenVector<T>::Flatten(*Out);
    auto place = context.GetEigenDevice<Place>();

    out.device(place) = x.abs().sum();
  }
};

// dX = dout * sign(X)
template <typename Place, typename T>
class L1NormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const framework::Tensor *X = context.Input<framework::Tensor>("X");
    const framework::Tensor *d_out =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    PADDLE_ENFORCE(d_out->numel() == 1, "L1 Norm Gradient should be scalar");
    framework::Tensor *dX =
        context.Output<framework::Tensor>(framework::GradVarName("X"));
    dX->mutable_data<T>(context.GetPlace());

    auto x = framework::EigenVector<T>::Flatten(*X);
    auto dout = framework::EigenVector<T>::Flatten(*d_out);
    auto dx = framework::EigenVector<T>::Flatten(*dX);
    auto place = context.GetEigenDevice<Place>();

    Eigen::DSizes<int, 1> x_dsize(X->numel());
    dx.device(place) = dout.broadcast(x_dsize) * x.sign();
  }
};

}  // namespace operators
}  // namespace paddle
