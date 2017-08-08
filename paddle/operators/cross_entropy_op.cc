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

#include "paddle/operators/cross_entropy_op.h"

namespace paddle {
namespace operators {

class OnehotCrossEntropyOp : public framework::OperatorWithKernel {
 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_EQ(ctx.InputSize(), 2,
                      "Input size of OnehotCrossEntropyOp must be two");
    PADDLE_ENFORCE_EQ(ctx.OutputSize(), 1,
                      "Output size of OnehotCrossEntropyOp must be one");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(0),
                            "0-th input of OnehotCrossEntropyOp should be set");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(1),
                            "1-th input of OnehotCrossEntropyOp should be set");
    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar(0),
                            "Outputs of OnehotCrossEntropyOp must all be set");
    PADDLE_ENFORCE_EQ(ctx.Input<Tensor>(0)->dims().size(), 2);
    PADDLE_ENFORCE_EQ(ctx.Output<Tensor>(0)->dims().size(), 1,
                      "label's dimension must be 1.");
    ctx.Output<Tensor>(0)->Resize({ctx.Input<Tensor>(0)->dims()[0]});
  }
};

class OnehotCrossEntropyGradientOp : public framework::OperatorWithKernel {
 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto X_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto X = ctx.Input<Tensor>("X");

    // TODO(superjom) add enforce here after helper functions ready
    X_grad->Resize(X->dims());
  }
};

class OnehotCrossEntropyOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  OnehotCrossEntropyOpMaker(framework::OpProto *proto,
                            framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The first input of OnehotCrossEntropyOp");
    AddInput("label", "The second input of OnehotCrossEntropyOp");
    AddOutput("Y", "The output of OnehotCrossEntropyOp");
    AddComment(R"DOC(
OnehotCrossEntropy Operator.

                Y[i] = -log(X[i][j])

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(onehot_cross_entropy, ops::OnehotCrossEntropyOp,
            ops::OnehotCrossEntropyOpMaker);
REGISTER_OP_CPU_KERNEL(
    onehot_cross_entropy,
    ops::OnehotCrossEntropyOpKernel<paddle::platform::CPUPlace, float>);
REGISTER_GRADIENT_OP(onehot_cross_entropy, onehot_cross_entropy_grad,
                     ops::OnehotCrossEntropyGradientOp);
REGISTER_OP_CPU_KERNEL(
    onehot_cross_entropy_grad,
    ops::OnehotCrossEntropyGradientOpKernel<paddle::platform::CPUPlace, float>);
