/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
Indicesou may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

class Unpool2dOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(Tensor) The input tensor of unpool operator. "
        "The format of input tensor is NCHW. Where N is batch size, C is the "
        "number of channels, H and W is the height and width of feature.");
    AddInput(
        "Indices",
        "(Tensor) The input tensor of the indices given out by MaxPool2d. "
        "The format of input tensor is NCHW. Where N is batch size, C is the "
        "number of channels, H and W is the height and width of feature.");
    AddOutput("Out",
              "(Tensor) The output tensor of unpool operator."
              "The format of output tensor is also NCHW."
              "Where N is batch size, C is "
              "the number of channels, H and W is the height and "
              "width of feature.");
    AddAttr<std::vector<int>>(
        "ksize",
        "(vector), the unpooling window size(height, width) "
        "of unpooling operator.");
    AddAttr<std::vector<int>>("strides",
                              "(vector, default:{1, 1}), "
                              "strides (height, width) of unpooling operator.")
        .SetDefault({1, 1});
    AddAttr<std::vector<int>>("paddings",
                              "(vector default:{0,0}), "
                              "paddings (height, width) of unpooling operator.")
        .SetDefault({0, 0});
    AddAttr<std::string>(
        "unpooling_type",
        "(string), unpooling type, can be \"max\" for max-unpooling ")
        .InEnum({"max"});
    AddAttr<std::vector<int>>("output_size",
                              "(vector, optional). The shape of output.")
        .SetDefault({0, 0})
        .SupportTensor();
    AddAttr<std::string>(
        "data_format",
        "(string, default NCHW) Only used in "
        "An optional string from: \"NHWC\", \"NCHW\". "
        "Defaults to \"NHWC\". Specify the data format of the output data, "
        "the input will be transformed automatically. ")
        .SetDefault("NCHW");
    AddComment(R"DOC(
Input shape is: $(N, C_{in}, H_{in}, W_{in})$, Output shape is:
$(N, C_{out}, H_{out}, W_{out})$, where
$$
H_{out} = (H_{in}-1) * strides[0] - 2 * paddings[0] + ksize[0] \\
W_{out} = (W_{in}-1) * strides[1] - 2 * paddings[1] + ksize[1]
$$
Paper: http://www.matthewzeiler.com/wp-content/uploads/2017/07/iccv2011.pdf
)DOC");
  }
};

int UnpoolOutputSize(int input_size, int ksize, int padding, int stride) {
  int output_size = (input_size - 1) * stride - 2 * padding + ksize;
  return output_size;
}

class UnpoolOp : public framework::OperatorWithKernel {
 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.GetPlace());
  }

 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

template <typename T>
class UnpoolOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Indices", this->Input("Indices"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

class UnpoolOpGrad : public framework::OperatorWithKernel {
 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.GetPlace());
  }

 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(unpool,
                            UnpoolInferShapeFunctor,
                            PD_INFER_META(phi::UnpoolInferMeta));
REGISTER_OPERATOR(unpool,
                  ops::UnpoolOp,
                  ops::Unpool2dOpMaker,
                  ops::UnpoolOpGradMaker<paddle::framework::OpDesc>,
                  ops::UnpoolOpGradMaker<paddle::imperative::OpBase>,
                  UnpoolInferShapeFunctor);

DECLARE_INFER_SHAPE_FUNCTOR(unpool_grad,
                            UnpoolGradInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

REGISTER_OPERATOR(unpool_grad, ops::UnpoolOpGrad, UnpoolGradInferShapeFunctor);
