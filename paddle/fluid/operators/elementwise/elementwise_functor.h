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

// Define the binary functors used in elementwise ops.
// Note: InverseXxxFunctor is needed when calling ElementwiseComputeEx on CPU.

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

// Floor Divide
template <typename T>
struct FloorDivFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    PADDLE_ENFORCE(b != 0, DIV_ERROR_INFO);
    return static_cast<T>(std::trunc(a / b));
  }
};

template <typename T>
struct InverseFloorDivFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    PADDLE_ENFORCE(a != 0, DIV_ERROR_INFO);
    return static_cast<T>(std::trunc(b / a));
  }
};

#undef DIV_ERROR_INFO

// Maximum
template <typename T>
struct MaxFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    return a > b ? a : b;
  }
};

// Minmum
template <typename T>
struct MinFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    return a < b ? a : b;
  }
};

// Mod
template <typename T, typename Enable = void>
struct ModFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    T res = a % b;

    // Accoding to PR#26732: in dividen % divsor
    // remainder shall have the same sign as divsor.
    return ((res != 0) && ((b ^ res) < 0)) ? res + b : res;
  }
};

template <typename T>
struct ModFunctor<T,
                  typename std::enable_if_t<std::is_floating_point<T>::value>> {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    T res = fmod(a, b);
    return ((res != 0) && ((res < 0) != (b < 0))) ? res + b : res;
  }
};

template <typename T, typename Enable = void>
struct InverseModFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    T res = b % a;
    res = ((res != 0) && ((a ^ res) < 0)) ? res + a : res;
    return res;
  }
};

template <typename T>
struct InverseModFunctor<
    T, typename std::enable_if_t<std::is_floating_point<T>::value>> {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    T res = fmod(b, a);
    return ((res != 0) && ((res < 0) != (a < 0))) ? res + a : res;
  }
};

}  // namespace operators
}  // namespace paddle
