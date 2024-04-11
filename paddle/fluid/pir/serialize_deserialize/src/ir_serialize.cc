// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/serialize_deserialize/include/ir_serialize.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/serialize_deserialize/include/serialize_utils.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/core/operation.h"

namespace pir {

Json ProgramWriter::GetProgramJson(const pir::Program* program) {
  program_json = WriteProgram(program);
  VLOG(6) << "Finish program to json.";
  return program_json;
}

Json ProgramWriter::WriteProgram(const pir::Program* program) {
  Json program_json;
  program_json[REGIONS] = Json::array();
  auto top_level_op = program->module_op();

  for (size_t i = 0; i < top_level_op->num_regions(); ++i) {
    std::string region_name = "region_" + std::to_string(region_id_++);
    auto& region = top_level_op->region(i);
    auto region_json = WriteRegion(&region, region_name);
    program_json[REGIONS].emplace_back(region_json);
  }
  VLOG(6) << "Finish write program.";
  return program_json;
}

Json ProgramWriter::WriteRegion(const pir::Region* region,
                                const std::string& region_name) {
  Json region_json;
  region_json[ID] = region_name;
  region_json[BLOCKS] = Json::array();
  for (auto block : region->blocks()) {
    std::string block_name = "block_" + std::to_string(block_id_++);
    auto block_json = WriteBlock(block, block_name);
    region_json[BLOCKS].emplace_back(block_json);
  }
  VLOG(6) << "Finish write " << region_name;
  return region_json;
}

Json ProgramWriter::WriteBlock(const pir::Block* block,
                               const std::string& block_name) {
  Json block_json;
  block_json[ID] = block_name;

  Json args_json = Json::array();
  for (auto arg : block->args()) {
    auto arg_json = WriteBlockArg(arg);
    args_json.emplace_back(arg_json);
  }
  block_json[BLOCKARGS] = args_json;

  Json ops_json = Json::array();
  for (auto op : block->ops()) {
    auto op_json = WriteOp(*op);
    ops_json.emplace_back(op_json);
  }
  block_json[BLOCKOPS] = ops_json;

  VLOG(6) << "Finish write " << block_name;
  return block_json;
}

Json ProgramWriter::WriteBlockArg(const pir::Value& value) {
  Json arg_json;
  Json var = WriteType(value.type());
  value_id_map[value] = blockarg_id_;
  arg_json[ID] = blockarg_id_;
  arg_json[TYPE_TYPE] = var;

  VLOG(6) << "Finish write blockargument " << blockarg_id_;
  blockarg_id_--;

  return arg_json;
}

Json ProgramWriter::WriteValue(const pir::Value& value) {
  Json var_json;
  // Json var = value;
  Json var = WriteType(value.type());
  value_id_map[value] = value_id_;
  var_json[ID] = value_id_;
  var_json[TYPE_TYPE] = var;
  VLOG(6) << "Finish write value " << value_id_;

  value_id_++;
  return var_json;
}

Json ProgramWriter::WriteOp(const pir::Operation& op) {
  Json op_json = Json::object();
  op_json[ID] = op.name();
  // serialize opoperands
  Json operands_json = Json::array();
  for (auto operand : op.operands()) {
    auto operand_json = WriteOpOperand(operand);
    operands_json.emplace_back(operand_json);
  }
  op_json[OPOPERANDS] = operands_json;

  // serialize opresults
  Json opresults_json = Json::array();
  for (auto& opresult : op.results()) {
    auto opresult_json = WriteValue(opresult);
    opresults_json.emplace_back(opresult_json);
  }
  op_json[OPRESULTS] = opresults_json;

  // serialize attributes
  op_json[ATTRS] = WriteAttributesMapOpinfo(const_cast<pir::Operation*>(&op),
                                            op.attributes());
  if (trainable_) {
    op_json[OPRESULTS_ATTRS] = WriteAttributesMapOther(op.attributes());
  }

  VLOG(6) << "Finish write Operation " << op.name();
  return op_json;
}

Json ProgramWriter::WriteOpOperand(const pir::OpOperand& op_operand) {
  Json operand_json = Json::object();
  int64_t id = value_id_map[op_operand.source()];
  operand_json[ID] = id;
  VLOG(6) << "Finish write OpOperand " << id;
  return operand_json;
}

Json ProgramWriter::WriteAttributesMapOpinfo(pir::Operation* op,
                                             const AttributeMap& attr_map) {
  Json attrs_json = Json::array();

  if (op->dialect()->name() == "pd_op") {
    if (op->dyn_cast<paddle::dialect::OpYamlInfoInterface>() != nullptr) {
      auto [_1, attr_info, _3, _4, _5] =
          op->dyn_cast<paddle::dialect::OpYamlInfoInterface>().GetOpInfo();
      if (attr_info.size() != 0) {
        for (auto it = attr_info.begin(); it != attr_info.end(); it++) {
          if (attr_map.find(it->name) != attr_map.end()) {
            attrs_json.emplace_back(
                WriteAttribute(it->name, attr_map.at(it->name)));
          }
        }
      }
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "the %s do not has OpYamlInfoInterface", op->name()));
    }
  } else {
    for (auto& attr : attr_map) {
      if (attr.first != "stop_gradient" && attr.first != "persistable" &&
          attr.first != "op_callstack") {
        attrs_json.emplace_back(WriteAttribute(attr.first, attr.second));
      }
    }
  }

  VLOG(6) << "Finish write Opinfo AttributeMap ";
  return attrs_json;
}

Json ProgramWriter::WriteAttributesMapOther(const AttributeMap& attr_map) {
  Json operesult_attrs_json = Json::array();
  for (auto& attr : attr_map) {
    if (attr.first == "stop_gradient" || attr.first == "persistable") {
      operesult_attrs_json.emplace_back(
          WriteAttribute(attr.first, attr.second));
    }
  }
  VLOG(6) << "Finish write Other AttributeMap ";
  return operesult_attrs_json;
}

Json ProgramWriter::WriteAttribute(const std::string& op_attr_name,
                                   const pir::Attribute& attr) {
  Json attr_json;
  attr_json[NAME] = op_attr_name;
  attr_json[ATTR_TYPE] = pir::writeAttr(attr);

  VLOG(6) << "Finish write Attribute. ";
  return attr_json;
}

Json ProgramWriter::WriteType(const pir::Type& type) {
  VLOG(6) << "Finish write Type. ";
  return pir::writeType(type);
}
}  // namespace pir
