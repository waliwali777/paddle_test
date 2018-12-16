/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#pragma once

#include <string>
#include "glog/logging.h"
#include "paddle/fluid/operators/jit/gen/act.h"
#include "paddle/fluid/operators/jit/gen/jitcode.h"

namespace paddle {
namespace operators {
namespace jit {
namespace gen {

class GRUJitCode : public VActJitCode {
 public:
  explicit GRUJitCode(int id, const gru_attr_t& attr, size_t code_size,
                      void* code_ptr = nullptr)
      : VActJitCode(attr.d, operand_type::sigmoid /* this is bugy*/, code_size,
                    code_ptr),
        id_(id) {
    auto typeExchange = [](KernelType type) -> gen::operand_type {
      if (type == KernelType::vsigmoid) {
        return operand_type::sigmoid;
      } else if (type == KernelType::vrelu) {
        return operand_type::relu;
      } else if (type == KernelType::vtanh) {
        return operand_type::tanh;
      } else if (type == KernelType::videntity) {
        return operand_type::identity;
      } else {
        LOG(FATAL) << "Do not support this jit::KernelType: " << type;
      }
      return operand_type::identity;
    };
    num_ = attr.d;
    act_gate_ = typeExchange(attr.act_gate);
    act_cand_ = typeExchange(attr.act_cand);

    this->genCode();
  }

  const char* name() const override {
    std::string base = "GRUJitCode";
    if (id_ == 0) {
      base += "_H1";
    } else if (id_ == 1) {
      base += "_HtPart1";
    } else if (id_ == 2) {
      base += "_HtPart2";
    }
    auto AddTypeStr = [&](operand_type type) {
      switch (type) {
        case operand_type::relu:
          base += "_Relu";
          break;
        case operand_type::exp:
          base += "_Exp";
          break;
        case operand_type::sigmoid:
          base += "_Sigmoid";
          break;
        case operand_type::tanh:
          base += "_Tanh";
          break;
        case operand_type::identity:
          base += "_Identity";
          break;
        default:
          break;
      }
    };
    AddTypeStr(act_gate_);
    AddTypeStr(act_cand_);
    return base.c_str();
  }
  void genCode() override;

 protected:
  int id_;
  int num_;
  operand_type act_gate_;
  operand_type act_cand_;
  reg64_t param1{abi_param1};
};

#define DECLARE_GRU_JITCODE(name, id)                                \
  class name##JitCode : public GRUJitCode {                          \
   public:                                                           \
    explicit name##JitCode(const gru_attr_t& attr, size_t code_size, \
                           void* code_ptr = nullptr)                 \
        : GRUJitCode(id, attr, code_size, code_ptr) {}               \
  };

DECLARE_GRU_JITCODE(GRUH1, 0);
DECLARE_GRU_JITCODE(GRUHtPart1, 1);
DECLARE_GRU_JITCODE(GRUHtPart2, 2);

#undef DECLARE_GRU_JITCODE

}  // namespace gen
}  // namespace jit
}  // namespace operators
}  // namespace paddle
