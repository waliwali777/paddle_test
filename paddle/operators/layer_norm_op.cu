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

#include "paddle/operators/elementwise_op_function.h"
#include "paddle/operators/layer_norm_op.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DataLayout = framework::DataLayout;

namespace {
template <typename T>
struct SubAndSquareFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return (a - b) * (a - b); }
};

template <typename T>
struct DivAndSqrtFunctor {
  explicit DivAndSqrtFunctor(T epsilon) { epsilon_ = epsilon; }
  inline HOSTDEVICE T operator()(T a, T b) const {
    return a / (sqrt(b) + epsilon_);
  }

 private:
  T epsilon_;
};

template <typename T>
struct MulFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a * b; }
};

template <typename T>
struct AddFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a + b; }
};

template <typename T>
struct SubFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a - b; }
};

template <typename T>
struct MulInvVarFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const {
    return a * std::sqrt(1.0 / b);
  }
};
}  // namespace

template <typename DeviceContext, typename T>
class LayerNormCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const float epsilon = ctx.Attr<float>("epsilon");
    auto *scale = ctx.Input<Tensor>("Scale");
    auto *bias = ctx.Input<Tensor>("Bias");
    auto x = *ctx.Input<Tensor>("X");

    auto *y = ctx.Output<Tensor>("Y");
    auto *mean = ctx.Output<Tensor>("Mean");
    auto *var = ctx.Output<Tensor>("Variance");
    const auto begin_norm_axis = ctx.Attr<int>("begin_norm_axis");

    const auto &x_dims = x.dims();

    y->mutable_data<T>(ctx.GetPlace());
    mean->mutable_data<T>(ctx.GetPlace());
    var->mutable_data<T>(ctx.GetPlace());

    auto matrix_dim = framework::flatten_to_2d(x_dims, begin_norm_axis);
    int left = static_cast<int>(matrix_dim[0]);
    int right = static_cast<int>(matrix_dim[1]);

    framework::DDim matrix_shape({left, right});

    x.Resize(matrix_shape);
    y->Resize(matrix_shape);

    auto &dev_ctx = ctx.template device_context<DeviceContext>();
    math::RowwiseMean<DeviceContext, T> row_mean;

    // functor-> get mean
    row_mean(dev_ctx, x, mean);

    // functor-> get variance
    ElementwiseComputeEx<SubAndSquareFunctor<T>, DeviceContext, T>(
        ctx, &x, mean, /*axis*/ 0, SubAndSquareFunctor<T>(), y);
    row_mean(dev_ctx, *y, var);

    // functor-> get norm_out
    ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
        ctx, &x, mean, /*axis*/ 0, SubFunctor<T>(), y);
    ElementwiseComputeEx<DivAndSqrtFunctor<T>, DeviceContext, T>(
        ctx, y, var, /*axis*/ 0, DivAndSqrtFunctor<T>(static_cast<T>(epsilon)),
        y);

    framework::DDim scale_shape({right});
    if (scale) {
      Tensor scale_matrix = *scale;
      scale_matrix.Resize(scale_shape);
      ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(
          ctx, y, &scale_matrix, /*axis*/ 1, MulFunctor<T>(), y);
    }
    if (bias) {
      Tensor bias_matrix = *bias;
      bias_matrix.Resize(scale_shape);
      ElementwiseComputeEx<AddFunctor<T>, DeviceContext, T>(
          ctx, y, &bias_matrix, /*axis*/ 1, AddFunctor<T>(), y);
    }
    y->Resize(x_dims);
  }
};

template <typename DeviceContext, typename T>
class LayerNormCUDAGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const float epsilon = ctx.Attr<float>("epsilon");
    auto x = *ctx.Input<Tensor>("X");
    auto mean = *ctx.Input<Tensor>("Mean");
    auto var = *ctx.Input<Tensor>("Variance");
    auto scale = *ctx.Input<Tensor>("Scale");
    auto d_y = *ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto begin_norm_axis = ctx.Attr<int>("begin_norm_axis");

    // init output
    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    const auto &x_dims = x.dims();
    auto matrix_dim = framework::flatten_to_2d(x_dims, begin_norm_axis);
    int left = static_cast<int>(matrix_dim[0]);
    int right = static_cast<int>(matrix_dim[1]);
    framework::DDim matrix_shape({left, right});

    d_y.Resize(matrix_shape);
    auto &dev_ctx = ctx.template device_context<DeviceContext>();
    math::ColwiseSum<DeviceContext, T> colwise_sum;

    Tensor temp;
    Tensor temp_norm;
    if (d_scale || d_x) {
      x.Resize(matrix_shape);
      temp.mutable_data<T>(matrix_shape, ctx.GetPlace());
      temp_norm.mutable_data<T>(matrix_shape, ctx.GetPlace());

      // get x_norm
      ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
          ctx, &x, &mean, /*axis*/ 0, SubFunctor<T>(), &temp_norm);
      ElementwiseComputeEx<DivAndSqrtFunctor<T>, DeviceContext, T>(
          ctx, &temp_norm, &var, /*axis*/ 0,
          DivAndSqrtFunctor<T>(static_cast<T>(epsilon)), &temp_norm);
    }

    if (d_bias) {
      d_bias->mutable_data<T>(ctx.GetPlace());
      colwise_sum(dev_ctx, d_y, d_bias);
    }
    if (d_scale) {
      d_scale->mutable_data<T>(ctx.GetPlace());
      ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(
          ctx, &temp_norm, &d_y, /*axis*/ 0, MulFunctor<T>(), &temp);
      colwise_sum(dev_ctx, temp, d_scale);
    }

    if (d_x) {
      framework::DDim vec_shape({left});
      d_x->mutable_data<T>(ctx.GetPlace());
      Tensor temp_vec;
      temp_vec.mutable_data<T>(vec_shape, ctx.GetPlace());

      auto &dev_ctx = ctx.template device_context<DeviceContext>();
      math::RowwiseMean<DeviceContext, T> row_mean;

      if (d_scale) {
        // dy_dx
        ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(
            ctx, &d_y, &scale, /*axis*/ 1, MulFunctor<T>(), &temp);
        framework::Copy(temp, ctx.GetPlace(), ctx.device_context(), d_x);

        // dy_dmean_dx
        row_mean(dev_ctx, temp, &temp_vec);
        ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
            ctx, d_x, &temp_vec, /*axis*/ 0, SubFunctor<T>(), d_x);

        // dy_var_dx
        ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(
            ctx, &temp, &temp_norm, /*axis*/ 0, MulFunctor<T>(), &temp);

      } else {
        // dy_dx
        framework::Copy(d_y, ctx.GetPlace(), ctx.device_context(), d_x);

        // dy_dmean_dx
        row_mean(dev_ctx, d_y, &temp_vec);
        ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
            ctx, d_x, &temp_vec, /*axis*/ 0, SubFunctor<T>(), d_x);

        // dy_var_dx
        ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(
            ctx, &d_y, &temp_norm, /*axis*/ 0, MulFunctor<T>(), &temp);
      }
      // dy_var_dx
      row_mean(dev_ctx, temp, &temp_vec);
      ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(
          ctx, &temp_norm, &temp_vec, /*axis*/ 0, MulFunctor<T>(), &temp_norm);
      ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
          ctx, d_x, &temp_norm, /*axis*/ 0, SubFunctor<T>(), d_x);

      ElementwiseComputeEx<DivAndSqrtFunctor<T>, DeviceContext, T>(
          ctx, d_x, &var, /*axis*/ 0,
          DivAndSqrtFunctor<T>(static_cast<T>(epsilon)), d_x);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    layer_norm,
    ops::LayerNormCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::LayerNormCUDAKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    layer_norm_grad,
    ops::LayerNormCUDAGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::LayerNormCUDAGradKernel<paddle::platform::CUDADeviceContext, double>);
