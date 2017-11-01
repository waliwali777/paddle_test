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

#include "paddle/operators/linear_chain_crf_op.h"

namespace paddle {
namespace operators {

class LinearChainCRFOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LinearChainCRFOpMaker(framework::OpProto* proto,
                        framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "Emission",
        "(LoDTensor, default: LoDTensor<float>). "
        "The unscaled emission weight matrix for the linear chain CRF. "
        "This input is a LoDTensor with shape [N x D] where N is the size of "
        "the mini-batch and D is the total tag number.");
    AddInput(
        "Transition",
        "(Tensor, default: Tensor<float>). A Tensor with shape [(D + 2) x D]. "
        "The learnable parameter for the linear_chain_crf operator. "
        "See more details in the operator's comments.");
    AddInput(
        "Label",
        "(LoDTensor, default: LoDTensor<int>). The ground truth which is a 2-D "
        "LoDTensor with shape [N x 1], where N is the total element number in "
        "a mini-batch.");
    AddOutput(
        "Alpha",
        "Tensor, default: Tensor<float>. The forward vectors for the entire "
        "batch. A two dimensional tensor with shape [N x D], "
        "denoted as \f$\alpha\f$. \f$\alpha$\f is a memo table used to "
        "calculate the normalization factor in CRF. \f$\alpha[k, v]$\f stores "
        "the unnormalized probabilites of all possible unfinished sequences of "
        "tags that end at position \f$k$\f with tag \f$v$\f. For each \f$k$\f, "
        "\f$\alpha[k, v]$\f is a vector of length \f$D$\f with a component for "
        "each tag value \f$v$\f. This vector is called a forward vecotr and "
        "will also be used in backward computations.")
        .AsIntermediate();
    AddOutput("EmissionExps",
              "The exponentials of Input(Emission). This is an intermediate "
              "computational result in forward computation, and will be reused "
              "in backward computation.")
        .AsIntermediate();
    AddOutput("TransitionExps",
              "The exponentials of Input(Transition). This is an intermediate "
              "computational result in forward computation, and will be reused "
              "in backward computation.")
        .AsIntermediate();
    AddOutput(
        "LogLikelihood",
        "(Tensor, default: Tensor<float>). The logarithm of the conditional "
        "likelihood of each training sample in a mini-batch. This is a 2-D "
        "tensor with shape [S x 1], where S is the sequence number in a "
        "mini-batch. Note: S is equal to the sequence number in a mini-batch. "
        "The output is no longer a LoDTensor.");
    AddComment(R"DOC(
Conditional Random Field defines an undirected probabilistic graph with nodes
denoting random variables and edges denoting dependencies between these
variables. CRF learns the conditional probability \f$P(Y|X)\f$, where
\f$X = (x_1, x_2, ... , x_n)\f$ are structured inputs and
\f$Y = (y_1, y_2, ... , y_n)\f$ are labels for the inputs.

Linear chain CRF is a special case of CRF that is useful for sequence labeling
task. Sequence labeling tasks do not assume a lot of conditional
independences among inputs. The only constraint they impose is that the input
and output must be linear sequences. Thus, the graph of such a CRF is a simple
chain or a line, which results in the linear chain CRF.

This operator implements the Forward-Backward algorithm for the linear chain
CRF. Please see http://www.cs.columbia.edu/~mcollins/fb.pdf and
http://cseweb.ucsd.edu/~elkan/250Bwinter2012/loglinearCRFs.pdf for reference.

Equation:

- Denote Input(Emission) to this operator as \f$x\f$ here.
- The first D values of Input(Transition) to this operator are for starting
weights, denoted as \f$a\f$ here.
- The next D values of Input(Transition) of this operator are for ending
weights, denoted as \f$b\f$ here.
- The remaning values of Input(Transition) are for transition weights,
denoted as \f$w\f$ here.
- Denote Input(Label) as \f$s\f$ here.

The probability of a sequence \f$s\f$ of length \f$L\f$ is defined as:
\f$P(s) = (1/Z) exp(a_{s_1} + b_{s_L}
                 + \sum_{l=1}^L x_{s_l}
                 + \sum_{l=2}^L w_{s_{l-1},s_l})\f$
where \f$Z\f$ is a normalization value so that the sum of \f$P(s)\f$ over
all possible sequences is \f$1\f$, and \f$x\f$ is the emission feature weight
to the linear chain CRF.

Finaly, the linear chain CRF operator outputs the logarithm of the conditional
likelihood of each training sample in a mini-batch.

NOTE:
1. The feature function for a CRF is made up of the emission features and the
transition features. The emission feature weights are NOT computed in
this operator. They MUST be computed first before this operator is called.

2. Because this operator performs global normalization over all possible
sequences internally, it expects UNSCALED emission feature weights.
Please do not call this op with the emission feature being output of any
nonlinear activation.

3. The 2nd dimension of Input(Emission) MUST be equal to the tag number.

)DOC");
  }
};

class LinearChainCRFOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Emission"),
                   "Input(Emission) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Transition"),
                   "Input(Transition) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");

    PADDLE_ENFORCE(ctx->HasOutput("Alpha"),
                   "Output(Alpha) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("EmissionExps"),
                   "Output(EmissionExps) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("TransitionExps"),
                   "Output(TransitionExps) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("LogLikelihood"),
                   "Output(LogLikelihood) should be not null.");

    auto emission_dims = ctx->GetInputDim("Emission");
    PADDLE_ENFORCE_EQ(emission_dims.size(), 2UL,
                      "The Input(Emission) should be a 2-D tensor.");
    PADDLE_ENFORCE(emission_dims[0], "An empty mini-batch is not allowed.");

    auto transition_dims = ctx->GetInputDim("Transition");
    PADDLE_ENFORCE_EQ(transition_dims.size(), 2UL,
                      "The Input(Transition) should be a 2-D tensor.");
    PADDLE_ENFORCE_EQ(
        transition_dims[0] - 2, transition_dims[1],
        "An invalid dimension for the Input(Transition), which should "
        "be a 2-D tensor with shape [(D + 2) x D].");
    PADDLE_ENFORCE_EQ(
        emission_dims[1], transition_dims[1],
        "The 2nd dimension of the Input(Emission) and the Input(Transition) "
        "should be equal to the tag number.");

    auto label_dims = ctx->GetInputDim("Label");
    PADDLE_ENFORCE(label_dims.size() == 2UL && label_dims[1] == 1UL,
                   "The Input(Label) should be a 2-D tensor with the 2nd "
                   "dimensions fixed to 1.");
    PADDLE_ENFORCE_EQ(
        emission_dims[0], label_dims[0],
        "The height of Input(Emission) and the height of Input(Label) "
        "should be the same.");

    ctx->SetOutputDim("Alpha", emission_dims);
    ctx->SetOutputDim("EmissionExps", emission_dims);
    ctx->SetOutputDim("TransitionExps", transition_dims);
    // TODO(caoying) This is tricky. The 1st dimension of Output(LogLikelihood)
    // is the sequence number in a mini-batch. The dimension set here should be
    // resized to its correct size in the function Compute. Fix this once we can
    // get LoD information in the InferShape interface.
    ctx->SetOutputDim("LogLikelihood", {emission_dims[0], 1});
  }

 protected:
  // Explicitly set that the data type of output of the linear_chain_crf
  // operator is determined by its input "Emission".
  framework::DataType IndicateDataType(
      const framework::ExecutionContext& ctx) const override {
    return framework::ToDataType(ctx.Input<LoDTensor>("Emission")->type());
  }
};

class LinearChainCRFGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("EmissionExps"),
                   "Input(EmissionExps) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("TransitionExps"),
                   "Input(TransitionExps) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("LogLikelihood")),
                   "Input(LogLikelihood@GRAD) shoudl be not null.");

    auto emission_exps_dims = ctx->GetInputDim("EmissionExps");
    PADDLE_ENFORCE_EQ(emission_exps_dims.size(), 2UL,
                      "The Input(EmissionExps) should be a 2-D tensor.");
    PADDLE_ENFORCE(emission_exps_dims[0],
                   "An empty mini-batch is not allowed.");

    auto transition_exps_dims = ctx->GetInputDim("TransitionExps");
    PADDLE_ENFORCE_EQ(transition_exps_dims.size(), 2UL,
                      "The Input(TransitionExps) should be a 2-D tensor.");
    PADDLE_ENFORCE_EQ(
        transition_exps_dims[0] - 2, transition_exps_dims[1],
        "An invalid dimension for the Input(TransitionExps), which should "
        "be a 2-D tensor with shape [(D + 2) x D].");
    PADDLE_ENFORCE_EQ(
        emission_exps_dims[1], transition_exps_dims[1],
        "The 2nd dimension of the Input(EmissionExps) and the "
        "Input(TransitionExps) should be equal to the tag number.");

    auto label_dims = ctx->GetInputDim("Label");
    PADDLE_ENFORCE(label_dims.size() == 2UL && label_dims[1] == 1UL,
                   "The Input(Label) should be a 2-D tensor with the 2nd "
                   "dimensions fixed to 1.");
    PADDLE_ENFORCE_EQ(
        emission_exps_dims[0], label_dims[0],
        "The height of Input(EmissionExps) and the height of Input(Label) "
        "should be the same.");

    if (ctx->HasOutput(framework::GradVarName("Emission"))) {
      ctx->SetOutputDim(framework::GradVarName("Emission"), emission_exps_dims);
    }
    if (ctx->HasOutput(framework::GradVarName("Transition"))) {
      ctx->SetOutputDim(framework::GradVarName("Transition"),
                        transition_exps_dims);
    }
  }

 protected:
  // Explicitly set that the data type of output of the linear_chain_crf_grad
  // operator is determined by its input: gradients of LogLikelihood.
  framework::DataType IndicateDataType(
      const framework::ExecutionContext& ctx) const override {
    return framework::ToDataType(
        ctx.Input<LoDTensor>(framework::GradVarName("LogLikelihood"))->type());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(linear_chain_crf, ops::LinearChainCRFOp, ops::LinearChainCRFOpMaker,
            linear_chain_crf_grad, ops::LinearChainCRFGradOp);
REGISTER_OP_CPU_KERNEL(
    linear_chain_crf,
    ops::LinearChainCRFOpKernel<paddle::platform::CPUPlace, float>,
    ops::LinearChainCRFOpKernel<paddle::platform::CPUPlace, double>);
REGISTER_OP_CPU_KERNEL(
    linear_chain_crf_grad,
    ops::LinearChainCRFGradOpKernel<paddle::platform::CPUPlace, float>,
    ops::LinearChainCRFGradOpKernel<paddle::platform::CPUPlace, double>);
