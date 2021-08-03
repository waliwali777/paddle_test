/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licnse. */

#include <memory>
#include <string>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class PowNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    auto factor = ctx.Attr<float>("factor");

    out->mutable_data<T>(ctx.GetPlace());

    const auto& runner = NpuOpRunner("Power", {*x}, {*out},
                                     {{"power", factor},
                                      {"scale", static_cast<float>(1.0)},
                                      {"shift", static_cast<float>(0.0)}});

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class PowGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto factor = ctx.Attr<float>("factor");

    auto x_dims = x->dims();

    auto place = ctx.GetPlace();
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // NOTE(liym27): dx = dout * factor * x.pow(factor-1)

    // Step1: Compute x_pow = x.pow(factor-1)
    Tensor x_pow(x->type());
    x_pow.mutable_data<T>(x->dims(), place);
    const auto& runner_pow = NpuOpRunner(
        "Power", {*x}, {x_pow}, {{"power", factor - static_cast<float>(1)}});
    runner_pow.Run(stream);

    // Step 2: Construct a broadcast factor, which has the same shape with x.

    // 2.1 Get a factor tensor with shape [1].
    Tensor factor_tensor(framework::proto::VarType::FP32);
    factor_tensor.mutable_data<float>({1}, place);
    FillNpuTensorWithConstant<float>(&factor_tensor, factor);

    // 2.2 Get the factor which has the shape with x and the same value with
    // factor.
    Tensor factor_bc_tensor(framework::proto::VarType::FP32);
    factor_bc_tensor.mutable_data<float>(x_dims, place);
    const auto& runner_bc =
        NpuOpRunner("FillD", {factor_tensor}, {factor_bc_tensor},
                    {{"dims", framework::vectorize(x_dims)}});
    runner_bc.Run(stream);

    // Step 3: Compute x_power_mul_factor = factor * x.pow(factor-1)
    Tensor x_power_mul_factor(x->type());
    x_power_mul_factor.mutable_data<T>(x->dims(), place);
    const auto& runner_mul_1 =
        NpuOpRunner("Mul", {factor_bc_tensor, x_pow}, {x_power_mul_factor}, {});
    runner_mul_1.Run(stream);

    // Step 4: Compute dx = dout * factor * x.pow(factor-1)
    dx->mutable_data<T>(place);
    const auto& runner_mul_2 =
        NpuOpRunner("Mul", {*dout, x_power_mul_factor}, {*dx}, {});
    runner_mul_2.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class ReluNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");

    out->mutable_data<T>(ctx.GetPlace());

    const auto& runner = NpuOpRunner("Relu",
                                     {
                                         *x,
                                     },
                                     {*out}, {});

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class ReluGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<Tensor>("Out");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    dx->mutable_data<T>(ctx.GetPlace());
    const auto& runner = NpuOpRunner("ReluGrad", {*dout, *out}, {*dx}, {});

    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class SqrtNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");

    auto* out = ctx.Output<Tensor>("Out");

    auto place = ctx.GetPlace();

    out->mutable_data<T>(place);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner = NpuOpRunner("Sqrt", {*x}, {*out}, {});
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class SqrtGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<Tensor>("Out");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto place = ctx.GetPlace();

    dx->mutable_data<T>(place);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner_dx = NpuOpRunner("SqrtGrad", {*out, *dout}, {*dx}, {});
    runner_dx.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class LogNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");

    auto* out = ctx.Output<Tensor>("Out");

    auto place = ctx.GetPlace();

    out->mutable_data<T>(place);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    Tensor one(x->type());
    one.mutable_data<T>(x->dims(), place);
    const auto& runner_one = NpuOpRunner("OnesLike", {*x}, {one}, {});
    runner_one.Run(stream);

    Tensor sub(x->type());
    sub.mutable_data<T>(x->dims(), place);
    const auto& runner_sub = NpuOpRunner("Sub", {*x, one}, {sub}, {});
    runner_sub.Run(stream);

    const auto& runner_out = NpuOpRunner("Log1p", {sub}, {*out}, {});
    runner_out.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class LogGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* x = ctx.Input<Tensor>("X");

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto place = ctx.GetPlace();

    dx->mutable_data<T>(place);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    const auto& runner = NpuOpRunner("DivNoNan", {*dout, *x}, {*dx}, {});
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class TanhNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");

    auto* out = ctx.Output<Tensor>("Out");

    auto place = ctx.GetPlace();

    out->mutable_data<T>(place);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner = NpuOpRunner("Tanh", {*x}, {*out}, {});
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class TanhGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* out = ctx.Input<Tensor>("Out");

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto place = ctx.GetPlace();

    dx->mutable_data<T>(place);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner_dx = NpuOpRunner("TanhGrad", {*out, *dout}, {*dx}, {});
    runner_dx.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class SquareNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");

    auto* out = ctx.Output<Tensor>("Out");

    auto place = ctx.GetPlace();

    out->mutable_data<T>(place);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner = NpuOpRunner("Square", {*x}, {*out}, {});
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class SigmoidNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");

    auto* out = ctx.Output<Tensor>("Out");

    auto place = ctx.GetPlace();

    out->mutable_data<T>(place);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner = NpuOpRunner("Sigmoid", {*x}, {*out}, {});
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class SigmoidGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* out = ctx.Input<Tensor>("Out");

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto place = ctx.GetPlace();

    dx->mutable_data<T>(place);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner_dx =
        NpuOpRunner("SigmoidGrad", {*out, *dout}, {*dx}, {});
    runner_dx.Run(stream);
  }
};

template <typename T>
void PrintTensor(const framework::Tensor& src,
                 const framework::ExecutionContext& ctx) {
  std::vector<T> vec(src.numel());
  TensorToVector(src, ctx.device_context(), &vec);
  for (int i = 0; i < static_cast<int>(vec.size()); ++i) {
    VLOG(4) << "vec[" << i << "] : " << vec[i];
  }
};

// HardSwish = min(max(0, x+offset), threshold) * x / scale
template <typename DeviceContext, typename T>
class HardSwishNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");

    float threshold = ctx.Attr<float>("threshold");
    float scale = ctx.Attr<float>("scale");
    float offset = ctx.Attr<float>("offset");

    auto place = ctx.GetPlace();

    out->mutable_data<T>(place);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    Tensor tensor_offset(x->type());
    tensor_offset.mutable_data<T>({1}, place);
    FillNpuTensorWithConstant<T>(&tensor_offset, static_cast<T>(offset));

    Tensor add_offset_val(x->type());
    add_offset_val.mutable_data<T>(x->dims(), place);
    const auto& runner_add =
        NpuOpRunner("AddV2", {*x, tensor_offset}, {add_offset_val});
    runner_add.Run(stream);

    Tensor zero_val(x->type());
    zero_val.mutable_data<T>(x->dims(), place);
    const auto& runner_zero = NpuOpRunner("ZerosLike", {*x}, {zero_val});
    runner_zero.Run(stream);

    Tensor max_val(x->type());
    max_val.mutable_data<T>(x->dims(), place);
    const auto& runner_max =
        NpuOpRunner("Maximum", {add_offset_val, zero_val}, {max_val});
    runner_max.Run(stream);

    Tensor tensor_threshold_tmp(x->type());
    tensor_threshold_tmp.mutable_data<T>({1}, place);
    FillNpuTensorWithConstant<T>(&tensor_threshold_tmp,
                                 static_cast<T>(threshold));
    Tensor tensor_threshold(x->type());
    tensor_threshold.mutable_data<T>(x->dims(), place);
    const auto& runner_fill =
        NpuOpRunner("FillD", {tensor_threshold_tmp}, {tensor_threshold},
                    {{"dims", framework::vectorize(x->dims())}});
    runner_fill.Run(stream);

    Tensor min_val(x->type());
    min_val.mutable_data<T>(x->dims(), place);
    const auto& runner_min =
        NpuOpRunner("Minimum", {max_val, tensor_threshold}, {min_val});
    runner_min.Run(stream);

    Tensor mul_val(x->type());
    mul_val.mutable_data<T>(x->dims(), place);
    const auto& runner_mul = NpuOpRunner("Mul", {*x, min_val}, {mul_val});
    runner_mul.Run(stream);

    const auto& runner_div =
        NpuOpRunner("Power", {mul_val}, {*out}, {{"scale", 1.0f / scale}});
    runner_div.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class HardSwishGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    float threshold = ctx.Attr<float>("threshold");
    float scale = ctx.Attr<float>("scale");
    float offset = ctx.Attr<float>("offset");

    auto place = ctx.GetPlace();

    dx->mutable_data<T>(place);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    Tensor tensor_offset(x->type());
    tensor_offset.mutable_data<T>({1}, place);
    FillNpuTensorWithConstant<T>(&tensor_offset, static_cast<T>(offset));

    Tensor add_offset_val(x->type());
    add_offset_val.mutable_data<T>(x->dims(), place);
    const auto& runner_add =
        NpuOpRunner("AddV2", {*x, tensor_offset}, {add_offset_val});
    runner_add.Run(stream);

    Tensor tensor_threshold_tmp(x->type());
    tensor_threshold_tmp.mutable_data<T>({1}, place);
    FillNpuTensorWithConstant<T>(&tensor_threshold_tmp,
                                 static_cast<T>(threshold));
    Tensor tensor_threshold(x->type());
    tensor_threshold.mutable_data<T>(x->dims(), place);
    const auto& runner_fill =
        NpuOpRunner("FillD", {tensor_threshold_tmp}, {tensor_threshold},
                    {{"dims", framework::vectorize(x->dims())}});
    runner_fill.Run(stream);

    Tensor tmp_bool1(framework::proto::VarType::BOOL);
    tmp_bool1.mutable_data<bool>(x->dims(), place);
    const auto& runner_less =
        NpuOpRunner("Less", {add_offset_val, tensor_threshold}, {tmp_bool1});
    runner_less.Run(stream);
    Tensor tmp1(x->type());
    tmp1.mutable_data<T>(x->dims(), place);
    auto dst_dtype = ConvertToNpuDtype(x->type());
    const auto& runner_cast1 =
        NpuOpRunner("Cast", {tmp_bool1}, {tmp1},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast1.Run(stream);

    Tensor zero_val(x->type());
    zero_val.mutable_data<T>(x->dims(), place);
    const auto& runner_zero = NpuOpRunner("ZerosLike", {*x}, {zero_val});
    runner_zero.Run(stream);

    Tensor tmp_bool2(framework::proto::VarType::BOOL);
    tmp_bool2.mutable_data<bool>(x->dims(), place);
    const auto& runner_greater =
        NpuOpRunner("Greater", {add_offset_val, zero_val}, {tmp_bool2});
    runner_greater.Run(stream);
    Tensor tmp2(x->type());
    tmp2.mutable_data<T>(x->dims(), place);
    const auto& runner_cast2 =
        NpuOpRunner("Cast", {tmp_bool2}, {tmp2},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast2.Run(stream);

    Tensor tmp3(x->type());
    tmp3.mutable_data<T>(x->dims(), place);
    const auto& runner_pow1 = NpuOpRunner("Power", {*x}, {tmp3},
                                          {{"scale", 2.0f}, {"shift", offset}});
    runner_pow1.Run(stream);

    Tensor tmp4(x->type());
    tmp4.mutable_data<T>(x->dims(), place);
    const auto& runner_mul1 = NpuOpRunner("Mul", {tmp1, tmp2}, {tmp4});
    runner_mul1.Run(stream);
    Tensor tmp5(x->type());
    tmp5.mutable_data<T>(x->dims(), place);
    const auto& runner_mul2 = NpuOpRunner("Mul", {tmp3, tmp4}, {tmp5});
    runner_mul2.Run(stream);

    Tensor tmp6(x->type());
    tmp6.mutable_data<T>(x->dims(), place);
    const auto& runner_pow2 = NpuOpRunner(
        "Power", {tmp5}, {tmp6}, {{"scale", 1.0f / scale}, {"shift", 1.0f}});
    runner_pow2.Run(stream);

    Tensor tmp7(x->type());
    tmp7.mutable_data<T>(x->dims(), place);
    const auto& runner_sub = NpuOpRunner("Sub", {tmp6, tmp1}, {tmp7});
    runner_sub.Run(stream);

    const auto& runner_final = NpuOpRunner("Mul", {tmp7, *dout}, {*dx});
    runner_final.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class HardSigmoidNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    float slope = ctx.Attr<float>("slope");
    float offset = ctx.Attr<float>("offset");

    out->mutable_data<T>(ctx.GetPlace());

    framework::NPUAttributeMap attr_input = {{"alpha", slope},
                                             {"beta", offset}};

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner = NpuOpRunner("HardSigmoid", {*x}, {*out}, attr_input);
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class HardSigmoidGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* out = ctx.Input<Tensor>("Out");

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    float slope = ctx.Attr<float>("slope");
    float offset = ctx.Attr<float>("offset");

    dx->mutable_data<T>(ctx.GetPlace());

    framework::NPUAttributeMap attr_input = {{"alpha", slope},
                                             {"beta", offset}};

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner_dx =
        NpuOpRunner("HardSigmoidGrad", {*dout, *out}, {*dx}, attr_input);
    runner_dx.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    pow, ops::PowNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::PowNPUKernel<paddle::platform::NPUDeviceContext,
                      paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    pow_grad, ops::PowGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::PowGradNPUKernel<paddle::platform::NPUDeviceContext,
                          paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    relu, ops::ReluNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ReluNPUKernel<paddle::platform::NPUDeviceContext,
                       paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    relu_grad,
    ops::ReluGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ReluGradNPUKernel<paddle::platform::NPUDeviceContext,
                           paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    sqrt, ops::SqrtNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::SqrtNPUKernel<paddle::platform::NPUDeviceContext,
                       paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    sqrt_grad,
    ops::SqrtGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::SqrtGradNPUKernel<paddle::platform::NPUDeviceContext,
                           paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    log, ops::LogNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::LogNPUKernel<paddle::platform::NPUDeviceContext,
                      paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    log_grad, ops::LogGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::LogGradNPUKernel<paddle::platform::NPUDeviceContext,
                          paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    tanh, ops::TanhNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::TanhNPUKernel<paddle::platform::NPUDeviceContext,
                       paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    tanh_grad,
    ops::TanhGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::TanhGradNPUKernel<paddle::platform::NPUDeviceContext,
                           paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    square, ops::SquareNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::SquareNPUKernel<paddle::platform::NPUDeviceContext,
                         paddle::platform::float16>,
    ops::SquareNPUKernel<paddle::platform::NPUDeviceContext, int>);

REGISTER_OP_NPU_KERNEL(
    sigmoid, ops::SigmoidNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::SigmoidNPUKernel<paddle::platform::NPUDeviceContext,
                          paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    sigmoid_grad,
    ops::SigmoidGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::SigmoidGradNPUKernel<paddle::platform::NPUDeviceContext,
                              paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    hard_swish,
    ops::HardSwishNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::HardSwishNPUKernel<paddle::platform::NPUDeviceContext,
                            paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    hard_swish_grad,
    ops::HardSwishGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::HardSwishGradNPUKernel<paddle::platform::NPUDeviceContext,
                                paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    hard_sigmoid,
    ops::HardSigmoidNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::HardSigmoidNPUKernel<paddle::platform::NPUDeviceContext,
                              paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    hard_sigmoid_grad,
    ops::HardSigmoidGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::HardSigmoidGradNPUKernel<paddle::platform::NPUDeviceContext,
                                  paddle::platform::float16>);
