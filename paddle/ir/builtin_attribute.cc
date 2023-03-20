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

#include "paddle/ir/builtin_attribute.h"

namespace ir {
std::string StrAttribute::data() const { return storage()->GetAsKey(); }

uint32_t StrAttribute::size() const { return storage()->GetAsKey().size(); }

NamedAttribute::NamedAttribute(StrAttribute name, Attribute value)
    : name_(name), value_(value) {}

StrAttribute NamedAttribute::name() const { return name_; }

Attribute NamedAttribute::value() const { return value_; }

void NamedAttribute::SetName(StrAttribute name) { name_ = name; }

void NamedAttribute::SetValue(Attribute value) { value_ = value; }

bool NamedAttribute::operator<(const NamedAttribute &right) const {
  return name().compare(right.name()) < 0;
}

bool NamedAttribute::operator==(const NamedAttribute &right) const {
  return name() == right.name() && value() == right.value();
}

bool NamedAttribute::operator!=(const NamedAttribute &right) const {
  return !(*this == right);
}

Attribute DictionaryAttribute::GetValue(const StrAttribute &name) {
  size_t left = 0;
  size_t right = storage()->size() - 1;
  size_t mid = 0;
  while (left <= right) {
    mid = (left + right) / 2;
    if (storage()->data()[mid].name().compare(name) < 0) {
      left = mid + 1;
    } else if (storage()->data()[mid].name().compare(name) > 0) {
      right = mid - 1;
    } else {
      return storage()->data()[mid].value();
    }
  }
  return nullptr;
}

uint32_t DictionaryAttribute::size() const { return storage()->size(); }
}  // namespace ir
