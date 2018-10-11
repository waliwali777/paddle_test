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


#include "mkldnn.hpp"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/operators/quantize_op.h"

namespace paddle {
namespace operators {

using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::reorder;
using platform::to_void_cast;
using Tensor = framework::Tensor;
using framework::DataLayout;
using mkldnn::stream;
using platform::GetMKLDNNFormat;

template <typename DeviceContext, typename T>
class QuantOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("Input");
    auto* scale = ctx.Input<Tensor>("Scale");
    auto* output = ctx.Output<Tensor>("Output");

    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& engine = dev_ctx.GetEngine();
 
    std::vector<primitive> pipeline;
    std::vector<int> src_tz = paddle::framework::vectorize2int(input->dims());
    std::vector<int> dst_tz = paddle::framework::vectorize2int(output->dims());

    const T* input_data = input->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    std::vector<T> scale_data = {*(scale->data<T>())};

    mkldnn::primitive_attr attri;
    int mask = 0;
    attri.set_output_scales(mask, scale_data);

    auto src_md = platform::MKLDNNMemDesc(
            {src_tz}, memory::data_type::f32, input->format());
    auto src_pd = mkldnn::memory::primitive_desc(src_md, engine);
    auto src_memory = std::make_shared<mkldnn::memory>(src_pd, to_void_cast<T>(input_data));
    std::shared_ptr<primitive::at> src_memory_p = std::shared_ptr<primitive::at>(new primitive::at(*src_memory));

    auto dst_md = platform::MKLDNNMemDesc(
            {dst_tz}, memory::data_type::u8, memory::format::nhwc);
    auto dst_pd = mkldnn::memory::primitive_desc(dst_md, engine);
    auto dst_memory = mkldnn::memory(dst_pd, to_void_cast<T>(output_data));
    
    auto reorder_pd = std::shared_ptr<reorder::primitive_desc>(
        new reorder::primitive_desc(dst_pd, src_pd, attri));    
    auto reorder_p= std::shared_ptr<reorder>(new reorder(*reorder_pd, *src_memory_p, dst_memory));
    pipeline.push_back(*reorder_p);
  }
};

framework::OpKernelType QuantOp::GetExpectedKernelType(const framework::ExecutionContext& ctx) const {
  framework::LibraryType library_{framework::LibraryType::kPlain};
  std::string data_format = ctx.Attr<std::string>("data_format");
  framework::DataLayout layout_ = framework::StringToDataLayout(data_format);
  if (library_ == framework::LibraryType::kPlain &&
      platform::CanMKLDNNBeUsed(ctx)) {
    library_ = framework::LibraryType::kMKLDNN;
    layout_ = framework::DataLayout::kMKLDNN;
  }
  return framework::OpKernelType(
      framework::ToDataType(ctx.Input<framework::LoDTensor>("Input")->type()),ctx.GetPlace(),layout_, library_);
      //ctx.device_context());
}


void QuantOpMaker::Make() {
  AddInput("Input","input");
  AddInput("Scale","scale...");
  AddOutput("Output","output");
  AddComment(R"DOC(
This op will quantize data from FP32 to INT8
)DOC");
}

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;


REGISTER_OPERATOR(quantize, ops::QuantOp, ops::QuantOpMaker, paddle::framework::DefaultGradOpDescMaker<true>);

REGISTER_OP_CPU_KERNEL(quantize, ops::QuantOpKernel<paddle::platform::CPUDeviceContext, float>);

//REGISTER_OP_KERNEL(quantization, MKLDNN, paddle::platform::CPUPlace, ops::QuantOpKernel<paddle::platform::CPUDeviceContext, float>);





