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

#include "paddle/fluid/translator/op_translator.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/translator/program_translator.h"
#include "paddle/fluid/translator/type_translator.h"
#include "paddle/ir/ir_context.h"
#include "paddle/ir/value.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace fluid {
namespace translator {

namespace {

using ResultIdx = size_t;
using OpDesc = paddle::framework::OpDesc;
using BlockDesc = paddle::framework::BlockDesc;
using VarDesc = paddle::framework::VarDesc;
using OpOutputTypeList = std::vector<ir::Type>;
using OpOutputMapping = std::unordered_map<std::string, ResultIdx>;

static const char kTargetDialectPrefix[] = "pd.";

inline bool IsInplace(const OpDesc& op_desc) {
  bool inplace = false;
  auto input_names = op_desc.InputArgumentNames();
  auto output_names = op_desc.OutputArgumentNames();

  std::vector<std::string> name_intersection;
  std::set_intersection(input_names.begin(),
                        input_names.end(),
                        output_names.begin(),
                        output_names.end(),
                        std::back_inserter(name_intersection));

  if (name_intersection.size() > 0) {
    std::string redundant_variables = std::accumulate(
        std::next(name_intersection.begin()),
        name_intersection.end(),
        name_intersection[0],
        [](std::string a, std::string b) { return a + "," + b; });
    VLOG(4) << "Following variables occur both in inputs and outputs: "
            << redundant_variables;
    return true;
  }

  return inplace;
}

inline ir::OpInfoImpl* LoopkUpOpInfo(ir::IrContext* ctx,
                                     const OpDesc& op_desc) {
  std::string target_op_name = kTargetDialectPrefix + op_desc.Type();
  if (IsInplace(op_desc)) {
    target_op_name += "_";
  }
  auto* op_info = ctx->GetRegisteredOpInfo(target_op_name);
  PADDLE_ENFORCE_NE(op_info,
                    nullptr,
                    platform::errors::PreconditionNotMet(
                        "Op %d should have corresponding OpInfo %d",
                        op_desc.Type(),
                        target_op_name));
  return op_info;
}

inline std::vector<ir::OpResult> GenerateOperationInput(
    TranslationContext* param_map, const OpDesc& op_desc) {
  std::vector<ir::OpResult> op_inputs = {};
  for (const auto& n : op_desc.Inputs()) {
    auto& name = n.first;
    VLOG(10) << "[input retriving]"
             << "[" << op_desc.Type() << "]" << name;
    auto& args = n.second;
    for (const auto& arg_name : args) {
      PADDLE_ENFORCE_NE(
          param_map->count(arg_name),
          0,
          platform::errors::PreconditionNotMet(
              "arg %s as input should be exists before prasing %d",
              arg_name,
              op_desc.Type()));
      op_inputs.push_back((*param_map)[arg_name]);
    }
  }
  return op_inputs;
}

inline std::tuple<OpOutputTypeList, OpOutputMapping> GenerateOperationOutput(
    ir::IrContext* ctx, const OpDesc& op_desc) {
  OpOutputMapping arg_to_idx;
  OpOutputTypeList op_output_types = {};

  auto& type_translator = TypeTranslator::instance();

  const BlockDesc* block = op_desc.Block();
  for (const auto& n : op_desc.Outputs()) {
    auto& name = n.first;
    VLOG(10) << "[output translating]"
             << "[" << op_desc.Type() << "]" << name;
    auto& args = n.second;
    for (const auto& arg_name : args) {
      VarDesc* var = block->FindVarRecursive(arg_name);
      VLOG(10) << "[output translating]"
               << "[" << op_desc.Type() << "]" << name << " " << arg_name << " "
               << var->GetType();

      ir::Type translated_var_type = type_translator[var->GetType()](ctx, *var);

      arg_to_idx[arg_name] = op_output_types.size();
      op_output_types.push_back(translated_var_type);
    }
  }
  return {op_output_types, arg_to_idx};
}

inline void RecordOpResultMapping(TranslationContext* param_map,
                                  const OpDesc& op_desc,
                                  ir::Operation* operation,
                                  const OpOutputMapping& arg_to_idx) {
  for (const auto& n : op_desc.Outputs()) {
    auto& name = n.first;
    VLOG(10) << "[output recording]"
             << "[" << op_desc.Type() << "]" << name;
    auto& args = n.second;
    for (const auto& arg_name : args) {
      auto idx = arg_to_idx.at(arg_name);
      VLOG(10) << "[output recording]"
               << "[" << op_desc.Type() << "]" << arg_name << " " << idx;

      (*param_map)[arg_name] = operation->GetResultByIndex(idx);
    }
  }
}

ir::Operation* GeneralOpHandler(ir::IrContext* ctx,
                                TranslationContext* param_map,
                                ir::Program* program,
                                const OpDesc& op_desc) {
  auto op_inputs = GenerateOperationInput(param_map, op_desc);

  OpOutputMapping arg_to_idx;
  OpOutputTypeList op_output_types = {};
  std::tie(op_output_types, arg_to_idx) = GenerateOperationOutput(ctx, op_desc);
  auto* op_info = LoopkUpOpInfo(ctx, op_desc);
  ir::Operation* operation =
      ir::Operation::create(op_inputs, op_output_types, {}, op_info);
  program->InsertOp(operation);
  RecordOpResultMapping(param_map, op_desc, operation, arg_to_idx);

  return operation;
}

ir::Operation* FeedOpHandler(ir::IrContext* ctx,
                             TranslationContext* param_map,
                             ir::Program* program,
                             const OpDesc& op_desc) {
  std::vector<ir::OpResult> op_inputs = {};

  OpOutputMapping arg_to_idx;
  OpOutputTypeList op_output_types = {};
  std::tie(op_output_types, arg_to_idx) = GenerateOperationOutput(ctx, op_desc);
  auto* op_info = LoopkUpOpInfo(ctx, op_desc);
  ir::Operation* operation =
      ir::Operation::create(op_inputs, op_output_types, {}, op_info);
  program->InsertOp(operation);
  RecordOpResultMapping(param_map, op_desc, operation, arg_to_idx);

  return operation;
}

ir::Operation* FetchOpHandler(ir::IrContext* ctx,
                              TranslationContext* param_map,
                              ir::Program* program,
                              const OpDesc& op_desc) {
  auto op_inputs = GenerateOperationInput(param_map, op_desc);

  OpOutputTypeList op_output_types = {};
  auto* op_info = LoopkUpOpInfo(ctx, op_desc);
  ir::Operation* operation =
      ir::Operation::create(op_inputs, op_output_types, {}, op_info);
  program->InsertOp(operation);

  return operation;
}
}  // namespace

OpTranslator::OpTranslator() : general_handler(GeneralOpHandler) {
  special_handlers["feed"] = FeedOpHandler;
  special_handlers["fetch_v2"] = FetchOpHandler;
}

}  // namespace translator
}  // namespace fluid
}  // namespace paddle
