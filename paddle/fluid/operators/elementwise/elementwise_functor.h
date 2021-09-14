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

#pragma once

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace operators {

// Add
template <typename T>
struct AddFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return a + b; }
};
template <typename T>
struct InverseAddFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return b + a; }
};

// Subtract
template <typename T>
struct SubFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return a - b; }
};
template <typename T>
struct InverseSubFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return b - a; }
};

// Multiply
template <typename T>
struct MulFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return a * b; }
};
template <typename T>
struct InverseMulFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return b * a; }
};

// Divide
#define DIV_ERROR_INFO                                             \
  "InvalidArgumentError: Integer division by zero encountered in " \
  "(floor) divide. Please check the input value."

template <typename T, typename Enable = void>
struct DivFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return a / b; }
};

template <typename T>
struct DivFunctor<T,
                  typename std::enable_if<std::is_integral<T>::value>::type> {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    // For int32/int64, need to check whether the divison is zero.
    PADDLE_ENFORCE(b != 0, DIV_ERROR_INFO);
    return a / b;
  }
};

template <typename T, typename Enable = void>
struct InverseDivFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return b / a; }
};

// Maximum
template <typename T>
struct MaxFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a > b ? a : b; }
};

// Minmum
template <typename T>
struct MinFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a < b ? a : b; }
};

// Pow
template <typename T>
struct PowFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
// TODO(wujionghao): A potential speed improvement is supporting different
// types in C++.
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
    // On CUDAPlace, std::pow(3, 1) calls pow(float, float), and
    // it will return a float number like 2.99... , which floor to 2
    // when cast to int by default and it is wrong.
    // Use llrint to cast it to the nearest integer, which is 3.
    if (std::is_integral<T>::value) {
      return std::llrint(std::pow(a, b));
    }
#endif
    return std::pow(a, b);
  }
};

}  // namespace operators
}  // namespace paddle
