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

#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename T>
class FillConstantMKLDNNHandler
    : public platform::MKLDNNHandlerNoCachingT<T, dnnl::binary> {
 public:
  FillConstantMKLDNNHandler(Tensor* out, const float value, dnnl::engine engine,
                            platform::Place cpu_place)
      : platform::MKLDNNHandlerNoCachingT<T, dnnl::binary>(engine, cpu_place) {
    const auto src0_md =
        dnnl::memory::desc({out->numel()}, platform::MKLDNNGetDataType<T>(),
                           dnnl::memory::format_tag::a);

    dnnl::primitive_attr attrs;
    dnnl::post_ops post_ops;

    post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, 0.0f, value);
    attrs.set_post_ops(post_ops);

    // we use binary_eq + linear combination to achieve set_constant behavior
    // binary_eq is done because operation NaN = X is always false, even if X is
    // also a NaN, that operation sets zeros to output memory, then linear
    // post-op sets desired constant in all out tensor. We cannot do only linear
    // activation, because newly allocated memory may contain NaN values and
    // adding to NaN results in another NaN
    this->AcquireForwardPrimitiveDescriptor(attrs, dnnl::algorithm::binary_eq,
                                            src0_md, src1_md, src0_md);
  }

  static const dnnl::memory::desc src1_md;
};

template <typename T>
const dnnl::memory::desc FillConstantMKLDNNHandler<T>::src1_md(
    {1}, platform::MKLDNNGetDataType<T>(), dnnl::memory::format_tag::a);

template <typename T>
class FillConstantMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    this->RunKernel(ctx);
  }

  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& dnnl_engine = dev_ctx.GetEngine();

    auto* out = ctx.Output<Tensor>("Out");

    T fill_value = CalculateFillValue(ctx);

    auto shape = GetShape(ctx);
    out->Resize(shape);

    FillConstantMKLDNNHandler<T> handler(out, fill_value, dnnl_engine,
                                         ctx.GetPlace());

    static const T nan =
        static_cast<T>(std::numeric_limits<float>::quiet_NaN());
    static dnnl::memory nan_memory = dnnl::memory(
        FillConstantMKLDNNHandler<T>::src1_md, mkldnn_engine, &nan);

    auto src0_memory_p = handler.AcquireDstMemory(out);
    auto fill_constant_p = handler.AcquireForwardPrimitive();

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    fill_constant_p->execute(astream, {{DNNL_ARG_SRC_0, *src0_memory_p},
                                       {DNNL_ARG_SRC_1, nan_memory},
                                       {DNNL_ARG_DST, *src0_memory_p}});
    astream.wait();

    out->set_layout(framework::DataLayout::kMKLDNN);
    out->set_format(platform::GetPlainMKLDNNFormat(out->dims().size()));
  }

  T CalculateFillValue(const framework::ExecutionContext& ctx) const {
    const auto str_value = ctx.Attr<std::string>("str_value");
    const auto float_value = ctx.Attr<float>("value");

    T value;

    if (str_value.empty()) {
      value = static_cast<T>(float_value);
    } else {
      // handle NaN/Inf first, which cannot be read from stream
      if (str_value == "inf") {
        value = static_cast<T>(std::numeric_limits<float>::infinity());
      } else if (str_value == "-inf") {
        value = static_cast<T>(-std::numeric_limits<float>::infinity());
      } else if (str_value == "nan") {
        value = static_cast<T>(std::numeric_limits<float>::quiet_NaN());
      } else {
        std::stringstream convert_stream(str_value);
        double tmp_value;
        convert_stream >> tmp_value;
        value = static_cast<T>(tmp_value);
      }
    }

    if (ctx.HasInput("ValueTensor")) {
      const auto* value_tensor = ctx.Input<Tensor>("ValueTensor");
      PADDLE_ENFORCE_EQ(
          value_tensor->numel(), 1,
          platform::errors::InvalidArgument(
              "When use Tensor as value to set Tensor value in fill_constant, "
              "value input(ValueTensor) size must be 1, but got %d",
              value_tensor->numel()));
      value = value_tensor->data<T>()[0];
    }

    return value;
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(fill_constant, MKLDNN, paddle::platform::CPUPlace,
                   ops::FillConstantMKLDNNKernel<float>);
