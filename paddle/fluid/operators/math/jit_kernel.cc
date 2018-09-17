/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/jit_kernel.h"
#include <functional>
#include <string>
#include "paddle/fluid/operators/math/cpu_vec.h"
#include "paddle/fluid/platform/cpu_info.h"

namespace paddle {
namespace operators {
namespace math {
namespace jitkernel {

KernelPool& KernelPool::Instance() {
  static KernelPool g_jit_kernels;
  return g_jit_kernels;
}

template <>
LSTMKernel<float>::LSTMKernel(int d, const std::string& act_gate_str,
                              const std::string& act_cand_str,
                              const std::string& act_cell_str)
    : Kernel(), d_(d) {
  if (platform::jit::MayIUse(platform::jit::avx512_common)) {
    math::VecActivations<float, platform::jit::avx512_common> act_functor;
    act_gate_ = act_functor(act_gate_str);
    act_cell_ = act_functor(act_cell_str);
    act_cand_ = act_functor(act_cand_str);
  } else if (platform::jit::MayIUse(platform::jit::avx2)) {
    math::VecActivations<float, platform::jit::avx2> act_functor;
    act_gate_ = act_functor(act_gate_str);
    act_cell_ = act_functor(act_cell_str);
    act_cand_ = act_functor(act_cand_str);
  } else if (platform::jit::MayIUse(platform::jit::avx)) {
    math::VecActivations<float, platform::jit::avx> act_functor;
    act_gate_ = act_functor(act_gate_str);
    act_cell_ = act_functor(act_cell_str);
    act_cand_ = act_functor(act_cand_str);
  } else {
    math::VecActivations<float, platform::jit::isa_any> act_functor;
    act_gate_ = act_functor(act_gate_str);
    act_cell_ = act_functor(act_cell_str);
    act_cand_ = act_functor(act_cand_str);
  }
}

template <>
const std::shared_ptr<LSTMKernel<float>>
KernelPool::Get<LSTMKernel<float>, int, const std::string&, const std::string&,
                const std::string&>(int d, const std::string& act_gate,
                                    const std::string& act_cand,
                                    const std::string& act_cell) {
  std::string key = "f" + std::to_string(d) + act_gate + act_cand + act_cell;
  if (kers_.find(key) == kers_.end()) {
    auto p =
        std::make_shared<LSTMKernel<float>>(d, act_gate, act_cand, act_cell);
    kers_.insert({key, std::dynamic_pointer_cast<Kernel>(p)});
    return p;
  }
  return std::dynamic_pointer_cast<LSTMKernel<float>>(kers_.at(key));
}

}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
