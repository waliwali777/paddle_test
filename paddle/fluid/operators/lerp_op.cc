// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/lerp_op.h"

namespace paddle {
namespace operators {

class LerpOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "lerp");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "lerp");
    OP_INOUT_CHECK(ctx->HasInput("Weight"), "Input", "Weight", "lerp");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "lerp");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    auto w_dims = ctx->GetInputDim("Weight");
    framework::DDim out_dims;
    out_dims = GetOutputDims(x_dims, y_dims);
    if (w_dims.size() > 1 || w_dims[0] != 1) {
      out_dims = GetOutputDims(out_dims, w_dims);
    }

    ctx->SetOutputDim("Out", out_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 private:
  framework::DDim GetOutputDims(const framework::DDim& x_dims,
                                const framework::DDim& y_dims) const {
    auto functor = [](const framework::DDim& small,
                      const framework::DDim& large) {
      std::vector<int64_t> shapes = framework::vectorize<int64_t>(large);
      for (int i = small.size() - 1, j = large.size() - 1; i >= 0; --i, --j) {
        int64_t s = small[i];
        int64_t l = large[j];
        if (s != l) {
          if (l == 1) {
            shapes[j] = s;
          } else if (s != 1) {
            PADDLE_THROW(platform::errors::InvalidArgument(
                "The shape of tensor a %s:%d must match shape of tensor b "
                "%s:%d.",
                small.to_str(), i, large.to_str(), j));
          }
        }
      }
      return framework::make_ddim(shapes);
    };
    if (x_dims.size() > y_dims.size()) {
      functor(y_dims, x_dims);
    }
    return functor(x_dims, y_dims);
  }
};

class LerpOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of lerp op.");
    AddInput("Y", "(Tensor), The input tensor of lerp op.");
    AddInput("Weight", "(Tensor, optional), The input tensor of lerp op.");
    AddOutput("Out", "(Tensor), The output tensor of lerp op.");
    AddComment(R"DOC(
Lerp Operator.

This operator is used to do a linear interpolation of input $X$ and $Y$ with $Weight$.

The equation is:

$$Out = X + Weight * (Y - X)$$

Both the input $X$ and $Y$ can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD information with input $X$.

)DOC");
  }
};

class LerpGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    }
    if (ctx->HasOutput(framework::GradVarName("Y"))) {
      ctx->SetOutputDim(framework::GradVarName("Y"), ctx->GetInputDim("Y"));
    }
  }
};

template <typename T>
class LerpOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    op->SetType("lerp_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("Weight", this->Input("Weight"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(
    lerp, paddle::operators::LerpOp, paddle::operators::LerpOpMaker,
    paddle::operators::LerpOpGradMaker<paddle::framework::OpDesc>,
    paddle::operators::LerpOpGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(lerp_grad, paddle::operators::LerpGradOp);

REGISTER_OP_CPU_KERNEL(
    lerp,
    paddle::operators::LerpKernel<paddle::platform::CPUDeviceContext, float>,
    paddle::operators::LerpKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    lerp_grad,
    paddle::operators::LerpGradKernel<paddle::platform::CPUDeviceContext,
                                      float>,
    paddle::operators::LerpGradKernel<paddle::platform::CPUDeviceContext,
                                      double>);
