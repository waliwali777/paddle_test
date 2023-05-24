// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ir/dialect.h"

namespace ir {
Dialect::Dialect(std::string name, ir::IrContext *context, ir::TypeId id)
    : name_(std::move(name)), context_(context), id_(id) {}

Dialect::~Dialect() = default;

void Dialect::RegisterInterface(std::unique_ptr<DialectInterface> interface) {
  VLOG(4) << "Register interface into dialect" << std::endl;
  auto it = registered_interfaces_.emplace(interface->interface_id(),
                                           std::move(interface));
  (void)it;
}

DialectInterface::~DialectInterface() = default;

IrContext *DialectInterface::ir_context() const {
  return dialect_->ir_context();
}

}  // namespace ir
