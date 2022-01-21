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

class CReduceMinOpMaker : public CReduceOpMaker {
 protected:
  std::string GetName() const override { return "Min"; }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_WITHOUT_GRADIENT(c_reduce_min, ops::CReduceOp,
                             ops::CReduceMinOpMaker);

REGISTER_OP_CPU_KERNEL(c_reduce_min,
                       ops::CReduceOpCPUKernel<ops::kRedMin, float>,
                       ops::CReduceOpCPUKernel<ops::kRedMin, double>,
                       ops::CReduceOpCPUKernel<ops::kRedMin, int>,
                       ops::CReduceOpCPUKernel<ops::kRedMin, int64_t>,
                       ops::CReduceOpCPUKernel<ops::kRedMin, plat::float16>);
