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

#pragma once

#include "paddle/ir/core/attribute.h"
#include "paddle/ir/core/builtin_attribute_storage.h"
#include "paddle/ir/core/utils.h"

namespace ir {
class IR_API StrAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(StrAttribute, StrAttributeStorage);

  bool operator<(const StrAttribute& right) const {
    return storage() < right.storage();
  }

  std::string data() const;

  uint32_t size() const;
};

class IR_API BoolAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(BoolAttribute, BoolAttributeStorage);

  bool data() const;
};

class IR_API FloatAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(FloatAttribute, FloatAttributeStorage);

  float data() const;
};

class IR_API DoubleAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(DoubleAttribute, DoubleAttributeStorage);

  double data() const;
};

class IR_API Int32Attribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(Int32Attribute, Int32AttributeStorage);

  int32_t data() const;
};

class IR_API Int64Attribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(Int64Attribute, Int64AttributeStorage);

  int64_t data() const;
};

class IR_API ArrayAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(ArrayAttribute, ArrayAttributeStorage);

  std::vector<Attribute> data() const;

  size_t size() const { return data().size(); }

  bool empty() const { return data().empty(); }

  Attribute operator[](size_t index) const { return data()[index]; }
};

class IR_API PointerAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(PointerAttribute, PointerAttributeStorage);

  void* data() const;
};

class IR_API TypeAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(TypeAttribute, TypeAttributeStorage);

  Type data() const;
};

}  // namespace ir

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::StrAttribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::BoolAttribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::FloatAttribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::DoubleAttribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::Int32Attribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::Int64Attribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::ArrayAttribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::PointerAttribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::TypeAttribute)
