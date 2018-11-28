// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace details {

/*
 * TODO(zjl): currently, we only consider two situation that output can
 * share memory with input, that is:
 *  - Input is never written after sharing, e.g. reshape_op, flatten_op
 *  - Input never appear after sharing, e.g. activation_ops (relu, scale...)
 *
 * In fact, more situations should be concerned, but more complicated.
 */
class InplaceOpPass : public ir::Pass {
 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(
      std::unique_ptr<ir::Graph> graph) const override;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
