/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/sample_logits_op.h"
#include "paddle/fluid/operators/math/sample_prob.h"

namespace paddle {
namespace operators {

class SampleLogitsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Logits",
             "(Tensor, default: Tensor<float>), The unscaled log probabilities "
             "which is a 2-D tensor with shape [N x K]. N is the batch_size, "
             "and K is the class number.");
    AddInput("Labels",
             "(Tensor) The ground truth which is a 2-D tensor. Labels is a "
             "Tensor<int64> with shape [N x NT], where NT is the number of"
             "true labels for each example.");
    AddInput("CustomizedSamples",
             "(Tensor, default: Tensor<int64_t>), A 2-D tensor with shape [N, "
             "NT + S],"
             " where N is the batch size, NT is the number of true labels "
             "and S is the number of negtive sample for each example."
             "The first NT elements of each row should be the same with true "
             "labels, "
             "followed by S custom negtive samples. This tensor"
             "is only used when use_customized_samples is true.")
        .AsDispensable();
    AddInput(
        "CustomizedProbabilities",
        "(Tensor, default: Tensor<float>), A 2-D tensor with shape [N, NT + S]."
        "The tensor has the same shape with CustomSamples,"
        "and each element represents probability of element in CustomSamples. "
        "This "
        "tensor is only used when use_customized_samples is true.")
        .AsDispensable();
    AddOutput("Samples",
              "(Tensor, default: Tensor<int64_t>), A 2-D tensor with shape [N, "
              "NT + S]."
              "The outputs value of sampler, including NT true lables and S "
              "negetive samples "
              "for each example. This will be used in"
              "backward calculation.")
        .AsIntermediate();
    AddOutput(
        "Probabilities",
        "(Tensor, default: Tensor<float>), A 2-D tensor with shape [N, NT + S]."
        "The probabilites of sampled positive and negtive labels.")
        .AsIntermediate();
    AddOutput("SampledLogits",
              "(Tensor, default: Tensor<float>), A 2-D tensor with shape"
              "[N, NT + S]. The outputs value of sampled logits, which will be"
              "used in backward propagation.")
        .AsIntermediate();
    AddOutput(
        "SampledLabels",
        "(Tensor, default: Tensor<int64>), A 2-D tensor. The sampled labels"
        "with shape [N, NT]. The tonsor contains hard labels as input to "
        " softmax op, that is 0, 1, …, NT-1 because of the first NT elements"
        " of Sampels are positive lables.");
    AddAttr<bool>(
        "use_customized_samples",
        "An indicator whether to use customized samples with probabilities, if "
        "True"
        "the operator will use customized samples and customized probabilities"
        "otherwise, the operator will generate them by itself.")
        .SetDefault(false);
    AddAttr<bool>(
        "uniq",
        "An indicator whether to sample non-repetitive negtive labels, if True"
        "the operator will sample negtive labels without replacement."
        "Otherwise, the operator will sample negtive labels with replacement.")
        .SetDefault(true);
    AddAttr<bool>(
        "remove_accidental_hits",
        "An indicator whether to remove accidental hits when samples hits true"
        "labels, the removal is implemented by subtracting the corresponding"
        "logits by float_max to subpress their softmax to be zero.")
        .SetDefault(true);
    AddAttr<int>("num_samples", "The number of negative samples.");
    AddAttr<int>("seed", "Random seed for generating samples").SetDefault(0);

    AddComment(R"DOC(
  """
  Computes sampled output training logits and labels suitable for implementing
  sampled softmax.        
  """

)DOC");
  }
};

class SampleLogitsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Logits"),
                   "Input(Logits) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Labels"),
                   "Input(Labels) should be not null.");

    PADDLE_ENFORCE(ctx->HasOutput("Samples"),
                   "Output(Samples) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("Probabilities"),
                   "Output(Probabilities) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("SampledLogits"),
                   "Output(SampledLogits) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("SampledLabels"),
                   "Output(SampledLabels) should be not null.");

    auto logits_dims = ctx->GetInputDim("Logits");
    auto labels_dims = ctx->GetInputDim("Labels");

    PADDLE_ENFORCE_EQ(
        logits_dims.size(), 2UL,
        "The logits of softmax_with_cross_entropy should be a 2-D tensor.");
    PADDLE_ENFORCE_EQ(labels_dims.size(), 2UL,
                      "The labels should be a 2-D tensor.");

    const int num_samples = ctx->Attrs().Get<int>("num_samples");
    const int num_sampled_classes = labels_dims[1] + num_samples;
    ctx->SetOutputDim("Samples", {logits_dims[0], num_sampled_classes});
    ctx->SetOutputDim("Probabilities", {logits_dims[0], num_sampled_classes});
    ctx->SetOutputDim("SampledLogits", {logits_dims[0], num_sampled_classes});
    ctx->SetOutputDim("SampledLabels", {logits_dims[0], labels_dims[1]});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("Logits"));
    framework::OpKernelType kt =
        framework::OpKernelType(data_type, ctx.device_context());
    return kt;
  }
};

// UNDERSTAND: InferShape for Grad
class SampleLogitsOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Logits"),
                   "Input(Logits) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Labels"),
                   "Input(Labels) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Samples"),
                   "Input(Samples) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("SampledLogits"),
                   "Input(SampledLogits) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("SampledLogits")),
                   "Input(SampledLogits@Grad) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Logits")),
                   "Output(Logits@Grad) should be not null.");

    auto logit_dims = ctx->GetInputDim("Logits");
    auto label_dims = ctx->GetInputDim("Labels");
    PADDLE_ENFORCE_EQ(label_dims.size(), 2UL,
                      "The label should be a 2-D tensor.");
    PADDLE_ENFORCE_EQ(logit_dims.size(), 2UL,
                      "The logits should be a 2-D tensor.");

    ctx->SetOutputDim(framework::GradVarName("Logits"),
                      ctx->GetInputDim("Logits"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(
        ctx.InputVar(framework::GradVarName("SampledLogits")));
    framework::OpKernelType kt =
        framework::OpKernelType(data_type, ctx.device_context());
    return kt;
  }
};

// UNDERSTAND: what's the rule for making a GradMaker TODO
class SampleLogitsGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto* grad_op = new framework::OpDesc();
    grad_op->SetType("sample_logits_grad");
    grad_op->SetInput("Logits", Input("Logits"));
    grad_op->SetInput("Labels", Input("Labels"));
    grad_op->SetInput("Samples", Output("Samples"));
    grad_op->SetInput("SampledLogits", Output("SampledLogits"));
    grad_op->SetInput(framework::GradVarName("SampledLogits"),
                      OutputGrad("SampledLogits"));
    grad_op->SetOutput(framework::GradVarName("Logits"), InputGrad("Logits"));
    grad_op->SetAttrMap(Attrs());
    return std::unique_ptr<framework::OpDesc>(grad_op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(sample_logits, ops::SampleLogitsOp, ops::SampleLogitsOpMaker,
                  ops::SampleLogitsGradMaker);
REGISTER_OPERATOR(sample_logits_grad, ops::SampleLogitsOpGrad);
REGISTER_OP_CPU_KERNEL(sample_logits, ops::SampleLogitsKernel<float>,
                       ops::SampleLogitsKernel<double>);
REGISTER_OP_CPU_KERNEL(sample_logits_grad, ops::SampleLogitsGradKernel<float>,
                       ops::SampleLogitsGradKernel<double>);
