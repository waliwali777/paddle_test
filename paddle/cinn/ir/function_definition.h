// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <memory>

#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace ir {

struct Specialization {
  Expr condition;
};

struct DefinitionContents;
struct FunctionContents;

/**
 * A Function definition which can either represent a init or an update
 * definition.
 */
class Definition {
 public:
  explicit Definition(const std::shared_ptr<DefinitionContents>& contents)
      : contents_(contents) {}

 private:
  std::shared_ptr<DefinitionContents> contents_;
};

}  // namespace ir
}  // namespace cinn
