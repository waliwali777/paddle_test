//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/pull_sparse_op_v2.h"

namespace paddle {
namespace operators {

class PullSparseV2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_GE(ctx->Inputs("Ids").size(), 1UL,
                      "Input(Ids) of PullSparseV2Op should not be null.");
    PADDLE_ENFORCE_GE(ctx->Outputs("Out").size(), 1UL,
                      "Output(Out) of PullSparseV2Op should not be null.");

    auto hidden_size = static_cast<uint32_t>(ctx->Attrs().Get<int>("EmbeddingDim"));
    auto all_ids_dim = ctx->GetInputsDim("Ids");
    const size_t n_ids = all_ids_dim.size();
    std::vector<framework::DDim> outs_dims;
    outs_dims.resize(n_ids);
    for (size_t i = 0; i < n_ids; ++i) {
      const auto ids_dims = all_ids_dim[i];
      int ids_rank = ids_dims.size();
      auto out_dim = framework::vectorize(ids_dims);
      out_dim.push_back(hidden_size);
      outs_dims[i] = framework::make_ddim(out_dim);
    }
    ctx->SetOutputsDim("Out", outs_dims);
    for (size_t i = 0; i < n_ids; ++i) {
      ctx->ShareLoD("Ids", "Out", i, i);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   ctx.device_context());
  }
};

class PullSparseV2OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Ids",
             "Input tensors with type int64 contains "
             "the ids to be looked up in PSLib. ")
        .AsDuplicable();
    AddInput("W", "The lookup table tensors.").AsDuplicable();
    AddOutput("Out", "The lookup results tensors.").AsDuplicable();
    AddAttr<int>("EmbeddingDim", "(int, the embedding hidden size").SetDefault(11);
    AddAttr<int>("TableId", "(int, the table id of this embedding").SetDefault(0);
    AddAttr<std::string>("AccessorClass", "(string, the class name of accessor").SetDefault("");
    AddAttr<std::string>("CtrLabelName", "(string, ctr label name").SetDefault("");
    AddAttr<int>("PaddingId", "(int, the padding id of this embedding").SetDefault(0);
    AddAttr<bool>("ScaleSparseGrad", "(bool, whether scale sparse gradient with batch size").SetDefault(true);
    AddAttr<bool>("AsyncPush", "(bool, whether push sparse is async").SetDefault(true);
    AddAttr<std::vector<std::string>>("InputNames", "(vector, slot names").SetDefault(std::vector<std::string>());
    AddComment(R"DOC(
Pull Sparse V2 Operator.

This operator is used to perform lookups on the PSLib
then concatenated into a dense tensor.

The input Ids can carry the LoD (Level of Details) information,
or not. And the output only shares the LoD information with input Ids.

)DOC");
  }
};

template <typename T>
class PushSparseV2OpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    std::unique_ptr<T> op(new T());
    op->SetType("push_sparse_v2");
    op->SetInput("Ids", this->Input("Ids"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("W", this->Input("W"));
    op->SetOutput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetAttrMap(this->Attrs());
    return op;
  }
};

class PushSparseV2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(pull_sparse_v2, ops::PullSparseV2Op,
                  ops::PullSparseV2OpMaker,
                  ops::PushSparseV2OpMaker<paddle::framework::OpDesc>,
                  ops::PushSparseV2OpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(push_sparse_v2, ops::PushSparseV2Op);
REGISTER_OP_CPU_KERNEL(pull_sparse_v2, ops::PullSparseV2CPUKernel<float>)
REGISTER_OP_CPU_KERNEL(push_sparse_v2, ops::PushSparseV2CPUKernel<float>)
