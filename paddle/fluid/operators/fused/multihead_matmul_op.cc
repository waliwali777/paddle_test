/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detail/safe_ref.h"

namespace paddle {
namespace operators {

class MultiHeadMatMulV2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE_EQ(context->HasInput("Input"), true,
                      "Input(Input) of MultiheadOp should not be null.");
    PADDLE_ENFORCE_EQ(context->HasInput("W"), true,
                      "Input(W) of MultiheadOp should not be null.");
    PADDLE_ENFORCE_EQ(context->HasInput("Bias"), true,
                      "Input(Bias) of MultiheadOp should not be null.");
    PADDLE_ENFORCE_EQ(context->HasInput("BiasQK"), true,
                      "Input(BiasQK) of MultiheadOp should not be null.");
    PADDLE_ENFORCE_EQ(context->HasOutput("Out"), true,
                      "Output(Out) of MatMulOp should not be null.");

    auto dim_w = context->GetInputDim("W");
    PADDLE_ENFORCE_GT(dim_w.size(), 2,
                      "Multihead input should be at least 3-D tensor.");

    auto dim_bias_q = context->GetInputDim("Bias");
    PADDLE_ENFORCE_GT(dim_bias_q.size(), 1,
                      "Multihead input should be at least 2-D tensor.");

    auto dim_bias_qk = context->GetInputDim("BiasQK");
    PADDLE_ENFORCE_GT(dim_bias_qk.size(), 3,
                      "Multihead input bias qk should be at least 4-D tensor.");

    int head_number = context->Attrs().Get<int>("head_number");
    PADDLE_ENFORCE_GT(head_number, 1,
                      "Multihead input head number should be at least 1.");
    // modify this
    auto dim_input = context->GetInputDim("Input");
    context->SetOutputDim("Out", dim_input);
    context->ShareLoD("Input", /*->*/ "Out");
  }
};

class MultiHeadMatMulV2OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "The input of MultiHeadMatMul op");
    AddInput("W", "The weight input of MultiHeadMatMul op");
    AddInput("Bias", "The bias input of MultiHeadMatMul op");
    AddInput("BiasQK", "The QK bias input of MultiHeadMatMul op");
    AddOutput("Out", "The output of MultiHeadMatMul op");
    AddAttr<bool>("transpose_Q",
                  R"DOC(If true, use the transpose of `Q`.
        )DOC")
        .SetDefault(false);
    AddAttr<bool>("transpose_K",
                  R"DOC(If true, use the transpose of `K`.
        )DOC")
        .SetDefault(true);
    AddAttr<bool>("transpose_V",
                  R"DOC(If true, use the transpose of `V`.
        )DOC")
        .SetDefault(false);
    AddAttr<float>("alpha", "The scale of Out").SetDefault(1.0f);
    AddAttr<int>("head_number", "The number of heads of the matrix")
        .SetDefault(1);
    AddComment(R"DOC(
MultiHeadMatMul Operator.

This op is used for optimize multi head calculation in ernie model.
Not suggest to use in other case except has same structure as ernie.

Example of matrix multiplication with head_number of B
- X: [B, M, K], Y: [B, K, N] => Out: [B, M, N]

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(multihead_matmul, ops::MultiHeadMatMulV2Op,
                             ops::MultiHeadMatMulV2OpMaker);
