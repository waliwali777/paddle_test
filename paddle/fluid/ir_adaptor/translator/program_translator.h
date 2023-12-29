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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/op_call_stack.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/core/value.h"

namespace paddle {
namespace translator {

struct VariableDefiningInfo {
  VariableDefiningInfo(pir::Value value,
                       bool generated_by_vector = false,
                       int idx_in_vector = -1)
      : value(value),
        generated_by_vector(generated_by_vector),
        idx_in_vector(idx_in_vector) {}
  VariableDefiningInfo() {}

  pir::Value value;

  bool generated_by_vector =
      false;  // true if target variable is generated by Vector<Tensor>
  int idx_in_vector =
      -1;  // positive if target variable is generated by Vector<Tensor>
};

class TranslationContext {
 public:
  using Key = std::string;
  using Value = VariableDefiningInfo;
  using ValueList = std::vector<Value>;
  using Container = std::unordered_map<Key, ValueList>;

  TranslationContext() {}
  explicit TranslationContext(TranslationContext* parent) : parent_(parent) {}
  ~TranslationContext() {}

  const Value& operator[](const Key& key) const;
  const Value& at(const Key& key) const;
  bool Has(const Key& key) const;
  size_t count(const Key& key)
      const;  // Caution: not exactly same as count in stl library

  void PushValue(const Key& key, const Value& value);
  void PopValue(const Key& key);
  TranslationContext* CreateInnerContext();

  Container::const_iterator begin() const { return container_.begin(); }
  Container::const_iterator end() const { return container_.end(); }

 private:
  Container container_;
  TranslationContext* parent_ = nullptr;
  std::vector<std::unique_ptr<TranslationContext>>
      sons_;  // used to seperate different block
};

class ProgramTranslator {
  using ProgramDesc = ::paddle::framework::ProgramDesc;
  using BlockDesc = ::paddle::framework::BlockDesc;
  using OpDesc = ::paddle::framework::OpDesc;
  using VarDesc = ::paddle::framework::VarDesc;

 public:
  explicit ProgramTranslator(const ProgramDesc* legacy_program,
                             pir::Program* program);

  void Translate();

  std::unordered_map<std::string, std::vector<pir::OpResult>>
  VarDesc2OpResult();

 private:
  const ProgramDesc* legacy_program_;  // not owned
  pir::Program* program_;              // not owned
  pir::IrContext* ctx_;                // not owned

  TranslationContext param_map_;
  std::unordered_map<std::string, VarDesc*> parameter_name_mappings_;
  std::unordered_set<std::string> parameter_visited_;

  /// In the legacy program desc, there are two special named varibales:
  /// 1. "feed", the input variable of feed op
  /// 2. "fetch", the output variable of fetch op
  /// However, new feed has no input and new fetch has no output
  /// So we don't handle these two vairables when
  /// `Get/SetParameterFromSingleBlock`
  static const std::unordered_set<std::string> no_cast_var_names;

  static const std::unordered_set<std::string> unsupported_ops;

  void TranslateBlock(const BlockDesc& src_block,
                      uint64_t start_id,
                      uint64_t end_id,
                      TranslationContext* translation_ctx,
                      pir::Block* dst_block);

  void TranslateGeneralOperation(const OpDesc* src_op,
                                 TranslationContext* translation_ctx,
                                 pir::Block* dst_block);

  void InsertDataOpForSingleBlock(const BlockDesc& block);
  void GetParameterForSingleBlock(const BlockDesc& block);
  void SetParameterFromSingleBlock(const BlockDesc& block);
  void SetStopGradientAttributeForAllValue(const BlockDesc& block);
  void SetIsPersisableAttributeForAllValue(const BlockDesc& block);

  const VariableDefiningInfo& GetValueOrCreateInTop(
      const std::string& var_name, TranslationContext* translation_ctx);

  const VariableDefiningInfo& CreateUndefinedVariable(
      const std::string& var_name, const BlockDesc& block);

  pir::Operation* InsertDataOpOrCreateArrayToBlock(pir::Block* insert_block,
                                                   pir::Type type);

  void TranslateIfOperation(const OpDesc* op,
                            TranslationContext* translation_ctx,
                            pir::Block* dst_block);

  void TranslateWhileOperation(const OpDesc* op,
                               TranslationContext* translation_ctx,
                               pir::Block* dst_block);
};

}  // namespace translator
}  // namespace paddle
