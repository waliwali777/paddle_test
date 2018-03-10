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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/softmax.h"
// #include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
// using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
// using DataLayout = platform::DataLayout;

// template <typename T>
// class SoftmaxCUDNNFunctor {
//  public:
//   void operator()(const platform::CUDADeviceContext& context,
//                   const framework::Tensor* X, framework::Tensor* Y);
// };

// template <typename T>
// void SoftmaxCUDNNFunctor<T>::operator()(
//     const platform::CUDADeviceContext& context, const framework::Tensor* X,
//     framework::Tensor* Y) {
//   // ------------------- cudnn descriptors ---------------------
//   ScopedTensorDescriptor xDesc;
//   ScopedTensorDescriptor yDesc;
//   DataLayout layout = DataLayout::kNCHW;

//   cudnnTensorDescriptor_t cudnn_x_desc =
//       xDesc.descriptor<T>(layout, framework::vectorize2int(X->dims()));
//   cudnnTensorDescriptor_t cudnn_y_desc =
//       xDesc.descriptor<T>(layout, framework::vectorize2int(Y->dims()));
//   // NOTE(*) The signature of cudnnSoftmaxForward
//   // final = alpha[0]*softmax + beta[0]*priorDstValue.
//   Tensor alpha, beta;
//   alpha.mutable_data<T>(X->dims(), context.GetPlace());
//   beta.mutable_data<T>(X->dims(), context.GetPlace());
//   // alpha.Resize(X->dims());
//   // beta.Resize(X->dims());
//   math::SetConstant<platform::CUDADeviceContext, T> constant;
//   constant(context, &alpha, static_cast<T>(1));
//   constant(context, &beta, static_cast<T>(0));

//   PADDLE_ENFORCE(platform::dynload::cudnnSoftmaxForward(
//       context.cudnn_handle(), CUDNN_SOFTMAX_FAST,
//       CUDNN_SOFTMAX_MODE_INSTANCE,
//       alpha.data<T>(), cudnn_x_desc, X->data<T>(), beta.data<T>(),
//       cudnn_y_desc,
//       Y->mutable_data<T>(context.GetPlace())));
// }

template <typename DeviceContext, typename T>
class SequenceSoftmaxCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<LoDTensor>("X");
    auto* out = ctx.Output<LoDTensor>("Out");

    auto lod = x->lod();
    auto dims = x->dims();

    const size_t level = lod.size() - 1;
    PADDLE_ENFORCE_EQ(dims[0], static_cast<int64_t>(lod[level].back()),
                      "The first dimension of Input(X) should be equal to the "
                      "sum of all sequences' lengths.");
    PADDLE_ENFORCE_EQ(dims[0], x->numel(),
                      "The width of each timestep in Input(X) of "
                      "SequenceSoftmaxOp should be 1.");

    out->mutable_data<T>(ctx.GetPlace());
    for (int i = 0; i < static_cast<int>(lod[level].size()) - 1; ++i) {
      int start_pos = static_cast<int>(lod[level][i]);
      int end_pos = static_cast<int>(lod[level][i + 1]);
      Tensor x_i = x->Slice(start_pos, end_pos);
      Tensor out_i = out->Slice(start_pos, end_pos);

      // Reshape from (end_pos - start_pos) x 1UL to 1UL x (end_pos - start_pos)
      framework::DDim dims_i = framework::make_ddim({1UL, end_pos - start_pos});
      x_i.Resize(dims_i);
      out_i.Resize(dims_i);
      math::SoftmaxCUDNNFunctor<T>()(
          ctx.template device_context<DeviceContext>(), &x_i, &out_i);
    }
  }
};

// template <typename DeviceContext, typename T>
// class SequenceSoftmaxGradKernel : public framework::OpKernel<T> {
//  public:
//   void Compute(const framework::ExecutionContext& ctx) const override {
//     auto* out = ctx.Input<LoDTensor>("Out");
//     auto* out_grad = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
//     auto* x = ctx.Input<LoDTensor>("X");
//     auto* x_grad = ctx.Output<LoDTensor>(framework::GradVarName("X"));

//     auto lod = x->lod();
//     const size_t level = lod.size() - 1;

//     x_grad->mutable_data<T>(ctx.GetPlace());
//     for (int i = 0; i < static_cast<int>(lod[level].size()) - 1; ++i) {

//       Tensor out_i = out->Slice(start_pos, end_pos);
//       Tensor out_grad_i = out_grad->Slice(start_pos, end_pos);
//       Tensor x_grad_i = x_grad->Slice(start_pos, end_pos);

//       // Reshape from (end_pos - start_pos) x 1UL to 1UL x (end_pos -
//       start_pos)
//       framework::DDim dims_i = framework::make_ddim({1UL, end_pos -
//       start_pos});
//       out_i.Resize(dims_i);
//       out_grad_i.Resize(dims_i);
//       x_grad_i.Resize(dims_i);
//       math::SoftmaxGradFunctor<DeviceContext, T>()(
//           ctx.template device_context<DeviceContext>(), &out_i, &out_grad_i,
//           &x_grad_i);
//     }
//   }
// };

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(
    sequence_softmax, CUDNN, ::paddle::platform::CUDAPlace,
    ops::SequenceSoftmaxCUDNNKernel<paddle::platform::CUDADeviceContext, float>)
