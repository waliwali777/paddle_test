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

#include "paddle/fluid/operators/fused/fused_ffn_op.h"
#include <algorithm>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

class FusedFfnOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "fused_ffn");
    OP_INOUT_CHECK(context->HasInput("Linear1Weight"), "Input", "Linear1Weight",
                   "fused_ffn");
    OP_INOUT_CHECK(context->HasInput("Linear2Weight"), "Input", "Linear2Weight",
                   "fused_ffn");
    OP_INOUT_CHECK(context->HasOutput("Out"), "Output", "Out", "fused_ffn");
    OP_INOUT_CHECK(context->HasOutput("Dropout1Mask"), "Output", "Dropout1Mask",
                   "fused_ffn");
    OP_INOUT_CHECK(context->HasOutput("Dropout2Mask"), "Output", "Dropout2Mask",
                   "fused_ffn");
    OP_INOUT_CHECK(context->HasOutput("Ln1Mean"), "Output", "Ln1Mean",
                   "fused_ffn");
    OP_INOUT_CHECK(context->HasOutput("Ln1Variance"), "Output", "Ln1Variance",
                   "fused_ffn");
    OP_INOUT_CHECK(context->HasOutput("Ln2Mean"), "Output", "Ln2Mean",
                   "fused_ffn");
    OP_INOUT_CHECK(context->HasOutput("Ln2Variance"), "Output", "Ln2Variance",
                   "fused_ffn");
    OP_INOUT_CHECK(context->HasOutput("Linear1Out"), "Output", "Linear1Out",
                   "fused_ffn");
    OP_INOUT_CHECK(context->HasOutput("Ln1Out"), "Output", "Ln1Out",
                   "fused_ffn");
    OP_INOUT_CHECK(context->HasOutput("Dropout1Out"), "Output", "Dropout1Out",
                   "fused_ffn");
    OP_INOUT_CHECK(context->HasOutput("Dropout2Out"), "Output", "Dropout2Out",
                   "fused_ffn");

    auto dim_x = context->GetInputDim("X");
    auto mat_dim_x =
        math::CreateMatrixDescriptor(RowMatrixFromVector(dim_x), 0, false);
    auto dim_Linear1Weight = context->GetInputDim("Linear1Weight");
    auto tmp_dim_x = dim_x;
    tmp_dim_x[dim_x.size() - 1] =
        dim_Linear1Weight[dim_Linear1Weight.size() - 1];
    context->SetOutputDim("Out", dim_x);
    if (context->Attrs().Get<bool>("is_test1") == false) {
      context->SetOutputDim("Dropout1Mask", tmp_dim_x);
    }
    context->SetOutputDim("Dropout1Out", tmp_dim_x);
    context->SetOutputDim("Linear1Out", tmp_dim_x);
    context->SetOutputDim("Ln1Out", dim_x);
    context->SetOutputDim("Dropout2Out", dim_x);

    if (context->Attrs().Get<bool>("is_test2") == false) {
      context->SetOutputDim("Dropout2Mask", dim_x);
    }
    context->SetOutputDim(
        "Ln1Mean",
        framework::make_ddim({mat_dim_x.batch_size_ * mat_dim_x.height_}));
    context->SetOutputDim(
        "Ln1Variance",
        framework::make_ddim({mat_dim_x.batch_size_ * mat_dim_x.height_}));
    context->SetOutputDim(
        "Ln2Mean",
        framework::make_ddim({mat_dim_x.batch_size_ * mat_dim_x.height_}));
    context->SetOutputDim(
        "Ln2Variance",
        framework::make_ddim({mat_dim_x.batch_size_ * mat_dim_x.height_}));

    context->ShareLoD("X", /*->*/ "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input = ctx.Input<Tensor>("X");
    auto input_data_type = input->type();
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const {
    if (framework::IsComplexType(expected_kernel_type.data_type_)) {
      // only promote inputs’s types when contains complex input
      return framework::OpKernelType(tensor.type(), tensor.place(),
                                     tensor.layout());
    } else {
      return framework::OpKernelType(expected_kernel_type.data_type_,
                                     tensor.place(), tensor.layout());
    }
  }
};

class FusedFfnOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of FusedFfn op");
    AddInput("Seed1",
             "The seed of dropout1 op, it has higher priority than the attr "
             "fix_seed and seed")
        .AsDispensable();
    AddInput("Seed2",
             "The seed of dropout2 op, it has higher priority than the attr "
             "fix_seed and seed")
        .AsDispensable();

    AddInput("Linear1Weight", "The linear1 weight of FusedFfn op");
    AddInput("Linear1Bias", "The linear1 bias of FusedFfn op").AsDispensable();
    AddInput("Linear2Weight", "The linear2 weight of FusedFfn op");
    AddInput("Linear2Bias", "The linear2 bias input of FusedFfn op")
        .AsDispensable();
    AddInput("Ln1Scale", "The layer_norm1 scale of FusedFfn op")
        .AsDispensable();
    AddInput("Ln1Bias", "The layer_norm1 bias of FusedFfn op").AsDispensable();
    AddInput("Ln2Scale", "The layer_norm2 scale of FusedFfn op")
        .AsDispensable();
    AddInput("Ln2Bias", "The layer_norm2 bias of FusedFfn op").AsDispensable();
    AddOutput("Out", "The output of FusedFfn op");
    AddOutput("Dropout1Mask", "The mask of dropout1").AsIntermediate();
    AddOutput("Dropout2Mask", "The mask of dropout2").AsIntermediate();
    AddOutput("Ln1Mean", "The mean of layer_norm1").AsIntermediate();
    AddOutput("Ln1Variance", "The variance of layer_norm1").AsIntermediate();
    AddOutput("Ln2Mean", "The mean of layer_nomr2").AsIntermediate();
    AddOutput("Ln2Variance", "The variance of layer_norm2").AsIntermediate();
    AddOutput("Linear1Out", "The output of linear1").AsIntermediate();
    AddOutput("Ln1Out", "The output of layer_norm1").AsIntermediate();
    AddOutput("Dropout1Out", "The output of dropout1").AsIntermediate();
    AddOutput("Dropout2Out", "The output of dropout2").AsIntermediate();

    AddAttr<bool>("normalize_pre_or_post", "true is pre layernorm");
    AddAttr<float>("epsilon1", "epsilon of layer_norm1").SetDefault(1e-8f);
    AddAttr<float>("epsilon2", "epsilon of layer_norm2").SetDefault(1e-8f);
    AddAttr<std::string>("act_method", "act_method").SetDefault("gelu");

    AddAttr<float>("dropout_prob1", "the dropout_prob of dropout1")
        .SetDefault(.5f)
        .AddCustomChecker([](const float &drop_p) {
          PADDLE_ENFORCE_EQ(
              drop_p >= 0.0f && drop_p <= 1.0f, true,
              platform::errors::InvalidArgument(
                  "'dropout_prob1' must be between 0.0 and 1.0."));
        });
    AddAttr<float>("dropout_prob2", "the dropout_prob of dropout2")
        .SetDefault(.5f)
        .AddCustomChecker([](const float &drop_p) {
          PADDLE_ENFORCE_EQ(
              drop_p >= 0.0f && drop_p <= 1.0f, true,
              platform::errors::InvalidArgument(
                  "'dropout_prob2' must be between 0.0 and 1.0."));
        });
    AddAttr<std::string>("dropout_implementation1",
                         "the dropout implementation of dropout1")
        .SetDefault("downgrade_in_infer")
        .AddCustomChecker([](const std::string &type) {
          PADDLE_ENFORCE_EQ(
              type == "downgrade_in_infer" || type == "upscale_in_train", true,
              platform::errors::InvalidArgument(
                  "dropout_implementation1 can only be downgrade_in_infer or "
                  "upscale_in_train"));
        });

    AddAttr<std::string>("dropout_implementation2",
                         "the dropout implementation of dropout2")
        .SetDefault("downgrade_in_infer")
        .AddCustomChecker([](const std::string &type) {
          PADDLE_ENFORCE_EQ(
              type == "downgrade_in_infer" || type == "upscale_in_train", true,
              platform::errors::InvalidArgument(
                  "dropout_implementation2 can only be downgrade_in_infer or "
                  "upscale_in_train"));
        });

    AddAttr<bool>("is_test1", "the is_test of dropout1").SetDefault(false);
    AddAttr<bool>("is_test2", "the is_test of dropout2").SetDefault(false);
    AddAttr<bool>("fix_seed1", "the is_test of dropout1").SetDefault(false);
    AddAttr<bool>("fix_seed2", "the is_test of dropout2").SetDefault(false);
    AddAttr<int>("seed1", "Dropout1 random seed.").SetDefault(0);
    AddAttr<int>("seed2", "Dropout2 random seed.").SetDefault(0);

    AddComment(R"DOC(
FusedFfn Operator.
)DOC");
  }
};

class FusedFfnOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->Attrs().Get<bool>("is_test1"), false,
                      platform::errors::InvalidArgument(
                          "GradOp is only callable when is_test1 is false"));
    PADDLE_ENFORCE_EQ(ctx->Attrs().Get<bool>("is_test2"), false,
                      platform::errors::InvalidArgument(
                          "GradOp is only callable when is_test2 is false"));
    OP_INOUT_CHECK(ctx->HasInput("Dropout1Mask"), "Input", "Dropout1Mask",
                   "FusedFfnGrad");
    OP_INOUT_CHECK(ctx->HasInput("Dropout2Mask"), "Input", "Dropout1Mask",
                   "FusedFfnGrad");
    OP_INOUT_CHECK(ctx->HasInput("Linear1Out"), "Input", "Linear1Out",
                   "FusedFfnGrad");
    OP_INOUT_CHECK(ctx->HasInput("Ln1Out"), "Input", "Ln1Out", "FusedFfnGrad");
    OP_INOUT_CHECK(ctx->HasInput("Dropout1Out"), "Input", "Dropout1Out",
                   "FusedFfnGrad");
    OP_INOUT_CHECK(ctx->HasInput("Dropout2Out"), "Input", "Dropout2Out",
                   "FusedFfnGrad");
    OP_INOUT_CHECK(ctx->HasInput("Linear1Weight"), "Input", "Linear1Weight",
                   "FusedFfnGrad");
    OP_INOUT_CHECK(ctx->HasInput("Linear2Weight"), "Input", "Linear2Weight",
                   "FusedFfnGrad");
    OP_INOUT_CHECK(ctx->HasInput("Ln1Mean"), "Input", "Ln1Mean",
                   "FusedFfnGrad");
    OP_INOUT_CHECK(ctx->HasInput("Ln1Variance"), "Input", "Ln1Variance",
                   "FusedFfnGrad");
    OP_INOUT_CHECK(ctx->HasInput("Ln2Mean"), "Input", "Ln2Mean",
                   "FusedFfnGrad");
    OP_INOUT_CHECK(ctx->HasInput("Ln2Variance"), "Input", "Ln2Variance",
                   "FusedFfnGrad");

    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "FusedFfnGrad");

    auto d_out_dim = ctx->GetInputDim(framework::GradVarName("Out"));
    ctx->SetOutputDim(framework::GradVarName("X"), d_out_dim);
    if (ctx->HasOutput(framework::GradVarName("Ln1Scale"))) {
      ctx->SetOutputDim(framework::GradVarName("Ln1Scale"),
                        ctx->GetInputDim("Ln1Scale"));
    }
    if (ctx->HasOutput(framework::GradVarName("Ln1Bias"))) {
      ctx->SetOutputDim(framework::GradVarName("Ln1Bias"),
                        ctx->GetInputDim("Ln1Bias"));
    }
    if (ctx->HasOutput(framework::GradVarName("Ln2Scale"))) {
      ctx->SetOutputDim(framework::GradVarName("Ln2Scale"),
                        ctx->GetInputDim("Ln2Scale"));
    }
    if (ctx->HasOutput(framework::GradVarName("Ln2Bias"))) {
      ctx->SetOutputDim(framework::GradVarName("Ln2Bias"),
                        ctx->GetInputDim("Ln2Bias"));
    }
    ctx->SetOutputDim(framework::GradVarName("Linear1Weight"),
                      ctx->GetInputDim("Linear1Weight"));
    if (ctx->HasOutput(framework::GradVarName("Linear1Bias"))) {
      ctx->SetOutputDim(framework::GradVarName("Linear1Bias"),
                        ctx->GetInputDim("Linear1Bias"));
    }
    ctx->SetOutputDim(framework::GradVarName("Linear2Weight"),
                      ctx->GetInputDim("Linear2Weight"));
    if (ctx->HasOutput(framework::GradVarName("Linear2Bias"))) {
      ctx->SetOutputDim(framework::GradVarName("Linear2Bias"),
                        ctx->GetInputDim("Linear2Bias"));
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input = ctx.Input<Tensor>("X");
    auto input_data_type = input->type();
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

template <typename T>
class FusedFfnOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("fused_ffn_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("X", this->Input("X"));
    op->SetInput("Linear1Weight", this->Input("Linear1Weight"));
    op->SetInput("Linear1Bias", this->Input("Linear1Bias"));
    op->SetInput("Linear2Weight", this->Input("Linear2Weight"));
    op->SetInput("Ln1Scale", this->Input("Ln1Scale"));
    op->SetInput("Ln1Bias", this->Input("Ln1Bias"));
    op->SetInput("Ln2Scale", this->Input("Ln2Scale"));
    op->SetInput("Ln2Bias", this->Input("Ln2Bias"));
    op->SetInput("Dropout1Mask", this->Output("Dropout1Mask"));
    op->SetInput("Dropout2Mask", this->Output("Dropout2Mask"));
    op->SetInput("Linear1Out", this->Output("Linear1Out"));
    op->SetInput("Ln1Out", this->Output("Ln1Out"));
    op->SetInput("Ln1Mean", this->Output("Ln1Mean"));
    op->SetInput("Ln1Variance", this->Output("Ln1Variance"));
    op->SetInput("Ln2Mean", this->Output("Ln2Mean"));
    op->SetInput("Ln2Variance", this->Output("Ln2Variance"));
    op->SetInput("Dropout1Out", this->Output("Dropout1Out"));
    op->SetInput("Dropout2Out", this->Output("Dropout2Out"));

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Ln1Scale"),
                  this->InputGrad("Ln1Scale"));
    op->SetOutput(framework::GradVarName("Ln1Bias"),
                  this->InputGrad("Ln1Bias"));
    op->SetOutput(framework::GradVarName("Ln2Scale"),
                  this->InputGrad("Ln2Scale"));
    op->SetOutput(framework::GradVarName("Ln2Bias"),
                  this->InputGrad("Ln2Bias"));
    op->SetOutput(framework::GradVarName("Linear1Weight"),
                  this->InputGrad("Linear1Weight"));
    op->SetOutput(framework::GradVarName("Linear1Bias"),
                  this->InputGrad("Linear1Bias"));
    op->SetOutput(framework::GradVarName("Linear2Weight"),
                  this->InputGrad("Linear2Weight"));
    if (this->HasInput("Linear2Bias")) {
      op->SetInput("Linear2Bias", this->Input("Linear2Bias"));
      op->SetOutput(framework::GradVarName("Linear2Bias"),
                    this->InputGrad("Linear2Bias"));
    }

    op->SetAttrMap(this->Attrs());
  }
};

template <typename T>
class FusedFfnOpDoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fused_ffn, ops::FusedFfnOp, ops::FusedFfnOpMaker,
                  ops::FusedFfnOpGradMaker<paddle::framework::OpDesc>,
                  ops::FusedFfnOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fused_ffn_grad, ops::FusedFfnOpGrad);
