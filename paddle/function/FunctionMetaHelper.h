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

#pragma once
#include <functional>
#include "BufferArg.h"
#include "BufferArgs.h"
#include "FunctionList.h"
#include "paddle/topology/Function.h"
#include "paddle/topology/meta/FunctionMeta.h"
#include "paddle/utils/Util.h"

namespace paddle {
namespace function {

typedef std::function<Error(const BufferArgs& ins,
                            const BufferArgs& outs,
                            const std::unordered_map<std::string, any>& attrs)>
    KernelTypeWithAttrs;

class FunctionMetaRegister {
public:
  FunctionMetaRegister(topology::meta::FunctionMetaPtr& meta) : meta_(meta) {}

  paddle::Error addCPUKernel(Function kernel) {
    return this->addKernel("CPUKernel", kernel);
  }

  paddle::Error addGPUKernel(Function kernel) {
    return this->addKernel("GPUKernel", kernel);
  }

  paddle::Error addCPUKernel(KernelTypeWithAttrs kernel) {
    return this->addKernel("CPUKernel", kernel);
  }

  paddle::Error addGPUKernel(KernelTypeWithAttrs kernel) {
    return this->addKernel("GPUKernel", kernel);
  }

private:
  template <typename T>
  paddle::Error addKernel(const std::string& name, T kernel) {
    return meta_->addMeta(name, kernel);
  }
  topology::meta::FunctionMetaPtr& meta_;
};

Function createKernel(const topology::Function& conf);

#define BEGIN_REGISTER_FUNCTION_META(name, func) \
  static paddle::InitFunction __init_##name##__([] {\
    paddle::topology::meta::FunctionMeta::registerFuncMeta(\
      #name, [](paddle::topology::meta::FunctionMetaPtr& meta) {\
  do {\
  function::FunctionMetaRegister reg(meta);\
  reg.addCPUKernel(func<DEVICE_TYPE_CPU>);\
  reg.addGPUKernel(func<DEVICE_TYPE_GPU>);\
} while(0);
#define END_REGISTER_FUNCTION_META() \
  return paddle::Error();            \
  }).check();                        \
  });

}  // namespace function
}  // namespace paddle
