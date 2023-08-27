/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "glog/logging.h"

#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/ftrl_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
namespace phi {

template <typename T, typename Context>
void FTRLOpSparseKernel(const Context& ctx,
                        const DenseTensor& grad,
                        const DenseTensor& param,
                        const DenseTensor& squared_accumulator,
                        const DenseTensor& learningRate,
                        float l1,
                        float l2,
                        float lr_power,
                        int64_t* rows,
                        int64_t row_numel,
                        DenseTensor* param_out,
                        DenseTensor* squared_accumulator_out,
                        DenseTensor* linear_accumulator_out) {
  auto g_ = phi::EigenVector<T>::Flatten(grad);
  auto p_ = phi::EigenVector<T>::Flatten(param);
  auto s_acc_ = phi::EigenVector<T>::Flatten(squared_accumulator);
  auto lr_ = phi::EigenVector<T>::Flatten(learningRate);
  auto l1_ = l1;
  auto l2_ = l2, auto lr_power_ = lr_power;
  auto rows_ = phi::EigenVector<int64_t>::Flatten(*rows);
  auto row_numel_ = row_numel;
  auto p_out_ = phi::EigenVector<T>::Flatten(*param_out);
  auto s_acc_out_ = phi::EigenVector<T>::Flatten(*squared_accumulator_out);
  auto l_acc_out_ = phi::EigenVector<T>::Flatten(*linear_accumulator_out);

  inline HOSTDEVICE void operator()(size_t i) {
    auto j = rows_[i / row_numel_] * row_numel_ + i % row_numel_;
    const T g = g_[i];
    const T p = p_[j];
    const T s_acc = s_acc_[j];
    const T lr = lr_[0];

    auto new_acc = s_acc + g * g;

    if (lr_power_ == static_cast<T>(-0.5)) {
      l_acc_out_[j] += g - (std::sqrt(new_acc) - std::sqrt(s_acc)) / lr * p;
    } else {
      l_acc_out_[j] +=
          g - (std::pow(new_acc, -lr_power_) - std::pow(s_acc, -lr_power_)) /
                  lr * p;
    }

    auto l_acc = l_acc_out_[j];

    if (std::fabs(l_acc) > l1_) {
      auto x = -l_acc;
      if (l_acc >= static_cast<T>(0)) {
        x += l1_;
      } else {
        x -= l1_;
      }

      auto y = static_cast<T>(2) * l2_;
      if (lr_power_ == static_cast<T>(-0.5)) {
        y += std::sqrt(new_acc) / lr;
      } else {
        y += std::pow(new_acc, -lr_power_) / lr;
      }

      auto pre_shrink = x / y;
      p_out_[j] = pre_shrink;
    } else {
      p_out_[j] = static_cast<T>(0);
    }

    s_acc_out_[j] += g * g;
  }
}

template <typename T, typename Context>
void FTRLOpKernel(const Context& ctx,
                  const DenseTensor& grad,
                  const DenseTensor& learningRate,
                  const DenseTensor& param,
                  const DenseTensor& squared_accumulator,
                  const DenseTensor& linear_accumulator,
                  float l1,
                  float l2,
                  float lr_power,
                  DenseTensor* param_out,
                  DenseTensor* squared_accumulator_out,
                  DenseTensor* linear_accumulator_out) {
  T l1_t = static_cast<T>(l1) + static_cast<T>(1e-10);
  T l2_t = static_cast<T>(l2) + static_cast<T>(1e-10);
  T lr_power_t = static_cast<T>(lr_power);
  auto g = phi::EigenVector<T>::Flatten(grad);
  auto p = phi::EigenVector<T>::Flatten(param);
  auto sq_accum = phi::EigenVector<T>::Flatten(squared_accumulator);
  auto lin_accum = phi::EigenVector<T>::Flatten(linear_accumulator);
  auto lr = phi::EigenVector<T>::Flatten(learningRate);

  auto p_out = phi::EigenVector<T>::Flatten(*param_out);
  auto s_acc_out = phi::EigenVector<T>::Flatten(*squared_accumulator_out);
  auto l_acc_out = phi::EigenVector<T>::Flatten(*linear_accumulator_out);
  auto& place = *ctx.eigen_device();

  Eigen::DSizes<int, 1> grad_dsize(grad.numel());

  auto new_accum = sq_accum + g * g;
  // Special case for lr_power_t = -0.5
  if (lr_power_t == static_cast<T>(-0.5)) {
    l_acc_out.device(place) =
        lin_accum + g -
        ((new_accum.sqrt() - sq_accum.sqrt()) / lr.broadcast(grad_dsize)) * p;
  } else {
    l_acc_out.device(place) =
        lin_accum + g -
        ((new_accum.pow(-lr_power_t) - sq_accum.pow(-lr_power_t)) /
         lr.broadcast(grad_dsize)) *
            p;
  }

  auto x_t = (l_acc_out.constant((l1_t)) * l_acc_out.sign() - l_acc_out);

  if (lr_power_t == static_cast<T>(-0.5)) {
    auto y_t = (new_accum.sqrt() / lr.broadcast(grad_dsize)) +
               l_acc_out.constant(static_cast<T>(2) * l2_t);
    auto pre_shrink = x_t / y_t;
    p_out.device(place) = (l_acc_out.abs() > l_acc_out.constant(l1_t))
                              .select(pre_shrink, p.constant(0));
  } else {
    auto y_t = (new_accum.pow(-lr_power_t) / lr.broadcast(grad_dsize)) +
               l_acc_out.constant(static_cast<T>(2) * l2_t);
    auto pre_shrink = x_t / y_t;
    p_out.device(place) = (l_acc_out.abs() > l_acc_out.constant(l1_t))
                              .select(pre_shrink, p.constant(0));
  }
  s_acc_out.device(place) = sq_accum + g * g;
}
}  // namespace phi
