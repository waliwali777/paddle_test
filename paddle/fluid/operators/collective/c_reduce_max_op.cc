/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/c_reduce_op.h"

namespace paddle {
namespace framework {
class OpDesc;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

class CReduceMaxOpMaker : public CReduceOpMaker {
 protected:
  std::string GetName() const override { return "Max"; }
};

DEFINE_C_REDUCE_CPU_KERNEL(CReduceMax, kRedMax)

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(c_reduce_max,
                             ops::CReduceOp,
                             ops::CReduceMaxOpMaker);

PD_REGISTER_STRUCT_KERNEL(c_reduce_max,
                          CPU,
                          ALL_LAYOUT,
                          ops::CReduceMaxCPUKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16) {}
