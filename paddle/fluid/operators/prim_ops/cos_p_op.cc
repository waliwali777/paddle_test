// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {
class CosPrimOp : public framework::OperatorBase {
 public:
  CosPrimOp(const std::string &type,
            const framework::VariableNameMap &inputs,
            const framework::VariableNameMap &outputs,
            const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Prim operator cos_p should not be excuted directly"));
  }
};

class CosPrimOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of cos_p op.");
    AddOutput("Y", "(Tensor), The output tensor of cos_p op.");
    AddComment(R"DOC(Autograd primitive cos_p operator.)DOC");
  }
};

class CosPrimOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    framework::InferShapeVarPtr x_var_ptr = ctx->GetInputVarPtrs("X")[0];
    framework::InferShapeVarPtr y_var_ptr = ctx->GetOutputVarPtrs("Y")[0];
    framework::VarDesc *x_var = PADDLE_GET(framework::VarDesc *, x_var_ptr);
    PADDLE_GET(framework::VarDesc *, y_var_ptr)->SetShape(x_var->GetShape());
  }
};

class CosPrimOpVarTypeInference
    : public framework::StaticGraphVarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto x_name = Input(ctx, "X")[0];
    auto y_name = Output(ctx, "Y")[0];
    SetType(ctx, y_name, GetType(ctx, x_name));
    SetDataType(ctx, y_name, GetDataType(ctx, x_name));
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(cos_p,
                  paddle::operators::CosPrimOp,
                  paddle::operators::CosPrimOpMaker,
                  paddle::operators::CosPrimOpShapeInference,
                  paddle::operators::CosPrimOpVarTypeInference);
