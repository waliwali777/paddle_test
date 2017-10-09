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

#include "paddle/operators/rank_loss_op.h"

namespace paddle {
namespace operators {

class RankLossOp : public framework::OperatorWithKernel {
 public:
  RankLossOp(const std::string &type, const framework::VariableNameMap &inputs,
             const framework::VariableNameMap &outputs,
             const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    // input check
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) shouldn't be null");
    PADDLE_ENFORCE(ctx->HasInput("Left"), "Input(Left) shouldn't be null");
    PADDLE_ENFORCE(ctx->HasInput("Right"), "Input(Right) shouldn't be null");

    auto label_dims = ctx->GetInputDim("Label");
    auto left_dims = ctx->GetInputDim("Left");
    auto right_dims = ctx->GetInputDim("Right");

    PADDLE_ENFORCE((label_dims == left_dims) && (left_dims == right_dims),
                   "All inputs must have the same size");
    PADDLE_ENFORCE((label_dims.size() == 2) && (label_dims[1] == 1),
                   "All inputs must be row vector with size batch_size x 1.");
    ctx->SetOutputDim("Out", label_dims);
  }
};

class RankLossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  RankLossOpMaker(framework::OpProto *proto,
                  framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Label",
             "The label indicating A ranked higher than B or not, row vector.");
    AddInput("Left", "The output of RankNet for doc A, vector.");
    AddInput("Right", "The output of RankNet for doc B, vetor");
    AddOutput("Out", "The output loss of RankLoss operator, vector.");
    AddComment(R"DOC(RankLoss operator

Rank loss operator for RankNet[1]. RankNet is a pairwise ranking model with
one training sample consisting of a pair of doc A and B, and the label P
indicating that A is ranked higher than B or not:

P = {0, 1} or {0, 0.5, 1}, where 0.5 means no information about the rank of
the input pair.

The RankLoss operator contains three inputs: Left (o_i), Right (o_j) and Label
(P_{i,j}), which represent the output of RankNet for two docs and the label
respectively, and yields the rank loss C_{i,j} by following the expression

\f[
  C_{i,j} = -\tilde{P_{ij}} * o_{i,j} + log(1 + e^{o_{i,j}}) \\
  o_{i,j} =  o_i - o_j  \\
  \tilde{P_{i,j}} = \left \{0, 0.5, 1 \right \} \ or \ \left \{0, 1 \right \}
\f]

The operator can take inputs of one sample or in batch.

[1]. Chris Burges, Tal Shaked, Erin Renshaw, et al. Learning to
     Rank using Gradient Descent.
     http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf
)DOC");
  }
};

class RankLossGradOp : public framework::OperatorWithKernel {
 public:
  RankLossGradOp(const std::string &type,
                 const framework::VariableNameMap &inputs,
                 const framework::VariableNameMap &outputs,
                 const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput("Left"), "Input(Left) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput("Right"), "Input(Right) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) shouldn't be null.");
    auto dims = ctx->GetInputDim("Left");
    auto left_grad_name = framework::GradVarName("Left");
    auto right_grad_name = framework::GradVarName("Right");

    if (ctx->HasOutput(left_grad_name)) {
      ctx->SetOutputDim(left_grad_name, dims);
    }

    if (ctx->HasOutput(right_grad_name)) {
      ctx->SetOutputDim(right_grad_name, dims);
    }
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OP(rank_loss, ops::RankLossOp, ops::RankLossOpMaker, rank_loss_grad,
            ops::RankLossGradOp);
REGISTER_OP_CPU_KERNEL(rank_loss,
                       ops::RankLossKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    rank_loss_grad, ops::RankLossGradKernel<paddle::platform::CPUPlace, float>);
