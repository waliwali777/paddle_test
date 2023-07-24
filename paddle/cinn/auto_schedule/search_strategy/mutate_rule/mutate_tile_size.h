// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/auto_schedule/search_strategy/mutate_rule/mutate_rule.h"

namespace cinn {
namespace auto_schedule {

/**
 * The rule to mutate tile size, witch will modify the factors of the Split
 * primitive.
 */
class MutateTileSize : public MutateRule {
 public:
  MutateTileSize() = default;

  ir::ScheduleDesc Apply(
      const ir::ScheduleDesc& trace,
      utils::LinearRandomEngine::StateType* rand_seed) override;
};

}  // namespace auto_schedule
}  // namespace cinn
