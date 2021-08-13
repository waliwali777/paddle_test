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
class Relu6NPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");

    out->mutable_data<T>(ctx.GetPlace());

    const auto& runner = NpuOpRunner("Relu6",
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
class Relu6GradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<Tensor>("Out");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    dx->mutable_data<T>(ctx.GetPlace());
    const auto& runner = NpuOpRunner("Relu6Grad", {*dout, *out}, {*dx}, {});

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
class LeakyReluNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    auto alpha = ctx.Attr<float>("alpha");

    out->mutable_data<T>(ctx.GetPlace());

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner =
        NpuOpRunner("LeakyRelu", {*x}, {*out}, {{"negative_slope", alpha}});
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class LeakyReluGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto alpha = ctx.Attr<float>("alpha");

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    dx->mutable_data<T>(ctx.GetPlace());
    const auto& runner = NpuOpRunner("LeakyReluGrad", {*dout, *x}, {*dx},
                                     {{"negative_slope", alpha}});

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

template <typename DeviceContext, typename T>
class ReciprocalNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    auto place = ctx.GetPlace();
    out->mutable_data<T>(place);
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    const auto& runner = NpuOpRunner("Reciprocal", {*x}, {*out}, {});
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class ReciprocalGradNPUKernel : public framework::OpKernel<T> {
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
    const auto& runner_dx =
        NpuOpRunner("ReciprocalGrad", {*out, *dout}, {*dx}, {});
    runner_dx.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class CosNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");

    auto place = ctx.GetPlace();
    out->mutable_data<T>(place);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner = NpuOpRunner("Cos", {*x}, {*out}, {});
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class CosGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* x = ctx.Input<Tensor>("X");
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto place = ctx.GetPlace();
    dx->mutable_data<T>(place);

    Tensor sin_out(x->type());  // Temporary Tensor
    sin_out.Resize(x->dims());
    sin_out.mutable_data<T>(place);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    const auto& runner = NpuOpRunner("Sin", {*x}, {sin_out}, {});
    runner.Run(stream);

    const auto& runner_dx = NpuOpRunner("Mul", {*dout, sin_out}, {*dx}, {});
    runner_dx.Run(stream);

    Tensor tmp(x->type());  // Temporary Tensor
    tmp.Resize(framework::make_ddim({1, 1}));
    tmp.mutable_data<T>(place);
    float factor = -1.;
    FillNpuTensorWithConstant<T>(&tmp, static_cast<T>(factor));

    const auto& runner_dx_ = NpuOpRunner("Xdivy", {*dx, tmp}, {*dx}, {});
    runner_dx_.Run(stream);
    // dx = -dout * Sine(x);
  }
};

template <typename DeviceContext, typename T>
class AtanNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    auto place = ctx.GetPlace();
    out->mutable_data<T>(place);
    const auto& runner = NpuOpRunner("Atan", {*x}, {*out}, {});
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class AtanGradNPUKernel : public framework::OpKernel<T> {
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
    const auto& runner_dx = NpuOpRunner("AtanGrad", {*x, *dout}, {*dx}, {});
    runner_dx.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class ExpNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* out = ctx.Output<framework::Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());
    const auto& runner = NpuOpRunner("Exp", {*x}, {*out}, {});
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class ExpGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<Tensor>("Out");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    const auto& runner = NpuOpRunner("Mul", {*dout, *out}, {*dx}, {});
    runner.Run(stream);
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
    relu6, ops::Relu6NPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::Relu6NPUKernel<paddle::platform::NPUDeviceContext,
                        paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    relu6_grad,
    ops::Relu6GradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::Relu6GradNPUKernel<paddle::platform::NPUDeviceContext,
                            paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    leaky_relu,
    ops::LeakyReluNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::LeakyReluNPUKernel<paddle::platform::NPUDeviceContext,
                            paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    leaky_relu_grad,
    ops::LeakyReluGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::LeakyReluGradNPUKernel<paddle::platform::NPUDeviceContext,
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
    hard_sigmoid,
    ops::HardSigmoidNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::HardSigmoidNPUKernel<paddle::platform::NPUDeviceContext,
                              paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    hard_sigmoid_grad,
    ops::HardSigmoidGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::HardSigmoidGradNPUKernel<paddle::platform::NPUDeviceContext,
                                  paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    reciprocal,
    ops::ReciprocalNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ReciprocalNPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::ReciprocalNPUKernel<paddle::platform::NPUDeviceContext,
                             paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    reciprocal_grad,
    ops::ReciprocalGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ReciprocalGradNPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::ReciprocalGradNPUKernel<paddle::platform::NPUDeviceContext,
                                 paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    cos, ops::CosNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::CosNPUKernel<paddle::platform::NPUDeviceContext,
                      paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    cos_grad, ops::CosGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::CosGradNPUKernel<paddle::platform::NPUDeviceContext,
                          paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    atan, ops::AtanNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::AtanNPUKernel<paddle::platform::NPUDeviceContext,
                       paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    atan_grad,
    ops::AtanGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::AtanGradNPUKernel<paddle::platform::NPUDeviceContext,
                           paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    exp, ops::ExpNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ExpNPUKernel<paddle::platform::NPUDeviceContext, double>);

REGISTER_OP_NPU_KERNEL(
    exp_grad, ops::ExpGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ExpGradNPUKernel<paddle::platform::NPUDeviceContext, double>);
