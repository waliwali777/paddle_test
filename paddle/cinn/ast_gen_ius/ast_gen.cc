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

#include "paddle/cinn/ast_gen_ius/ast_gen.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/operation.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_printer.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"

namespace cinn {
namespace ast_gen_ius {

ir::Expr ConvertReduceBody(ir::Expr body,
                           ir::Tensor tensor,
                           const std::vector<Expr>& axis_exprs) {
  ir::Reduce* reduce_node = body.As<ir::Reduce>();
  if (!reduce_node) {
    return ir::Store::Make(tensor, body, axis_exprs);
  }

  switch (reduce_node->reduce_type) {
    case ir::Reduce::kSum:
      return ir::Store::Make(
          tensor, tensor(axis_exprs) + reduce_node->body, axis_exprs);
    case ir::Reduce::kMul:
      return ir::Store::Make(
          tensor, tensor(axis_exprs) * reduce_node->body, axis_exprs);
    case ir::Reduce::kMax:
      return ir::Store::Make(
          tensor,
          ir::Max::Make(tensor(axis_exprs), reduce_node->body),
          axis_exprs);
    case ir::Reduce::kMin:
      return ir::Store::Make(
          tensor,
          ir::Min::Make(tensor(axis_exprs), reduce_node->body),
          axis_exprs);
    case ir::Reduce::kAll:
      return ir::Store::Make(
          tensor, tensor(axis_exprs) && reduce_node->body, axis_exprs);
    case ir::Reduce::kAny:
      return ir::Store::Make(
          tensor, tensor(axis_exprs) || reduce_node->body, axis_exprs);
    default:
      CINN_NOT_IMPLEMENTED
  }
}

ir::Expr AstGen::Build(const ir::Tensor& tensor, TensorGroup* tensor_group) {
  const std::vector<ir::Var>& axis = tensor->axis();
  const std::vector<ir::Expr>& shape = tensor->shape;
  size_t axis_len = axis.size();
  CHECK_EQ(shape.size(), axis_len) << "Internal Error: Tensor has different "
                                      "shape and axis length in AstGen";
  std::vector<ir::Expr> axis_exprs;
  for (const auto& a : axis) {
    axis_exprs.push_back(a);
  }

  if (tensor->is_reduce_tensor()) {
    // Make an init Tensor for domain without reduce axis
    Expr init_value = tensor->GetReduceInitVal();
    // TODO(zhhsplendid): Clean the handcoded "__reduce_init" string
    std::string reduce_init_name = tensor->name + "__reduce_init";
    const std::vector<Expr>& domain = tensor->domain_without_reduce_axis();
    ir::Tensor init_tensor = lang::Compute(
        domain,
        [=](const std::vector<Expr>& axis) { return init_value; },
        reduce_init_name);
    tensor_group->Insert(init_tensor);
    tensor_group->MarkShareMemBuffer(tensor, init_tensor);
    tensor_group->CtrlDepend(tensor, init_tensor);
    Expr init_body = ir::Store::Make(init_tensor, init_value, axis_exprs);
    // create schedule block itervars, i0,i1...
    std::vector<ir::Var> block_vars;
    for (int i = 0; i < shape.size(); ++i) {
      block_vars.push_back(Var(
          Expr(0), shape[i], cinn::UniqName("i" + std::to_string(i)), false));
      optim::ReplaceVarWithExpr(&init_body, axis[i], block_vars[i]);
    }
    init_body = ir::ScheduleBlockRealize::Make(
        axis_exprs,
        ir::ScheduleBlock::Make(
            block_vars, {}, {}, reduce_init_name, init_body));

    // For the remaining reduce axis, make reduce body
    const std::vector<ir::Var>& reduce_axis = tensor->reduce_axis;
    ir::Expr reduce_body =
        ConvertReduceBody(tensor->body(), tensor, axis_exprs);
    // create schedule block itervars, i0,i1...
    std::vector<ir::Var> reduce_block_vars(block_vars);
    std::vector<ir::Expr> reduce_block_exprs(axis_exprs);
    for (int i = 0; i < reduce_axis.size(); ++i) {
      int count = reduce_block_vars.size() + i;
      reduce_block_vars.push_back(
          Var(reduce_axis[i]->lower_bound,
              reduce_axis[i]->upper_bound,
              cinn::UniqName("i" + std::to_string(count)),
              false));
      reduce_block_exprs.push_back(reduce_axis[i]);
    }
    for (int i = 0; i < axis.size(); ++i) {
      optim::ReplaceVarWithExpr(&reduce_body, axis[i], reduce_block_vars[i]);
    }
    for (int i = axis.size(); i < reduce_block_vars.size(); ++i) {
      optim::ReplaceVarWithExpr(
          &reduce_body, reduce_axis[i - axis.size()], reduce_block_vars[i]);
    }

    reduce_body = ir::ScheduleBlockRealize::Make(
        reduce_block_exprs,
        ir::ScheduleBlock::Make(
            reduce_block_vars, {}, {}, tensor->name, reduce_body));
    for (int i = static_cast<int>(reduce_axis.size()) - 1; i >= 0; --i) {
      reduce_body = ir::For::Make(reduce_axis[i],
                                  reduce_axis[i]->lower_bound,
                                  reduce_axis[i]->upper_bound,
                                  ir::ForType::Serial,
                                  ir::DeviceAPI::Host,
                                  ir::Block::Make({reduce_body}));
    }

    // Put the two parts together
    ir::Expr body = ir::Block::Make({init_body, reduce_body});
    for (int i = static_cast<int>(axis_len) - 1; i >= 0; --i) {
      ir::Var loop_var = axis[i];
      ir::Expr loop_extent = shape[i];
      body = ir::For::Make(
          loop_var,
          Expr(0),
          loop_extent,
          ir::ForType::Serial,
          ir::DeviceAPI::Host,
          i == static_cast<int>(axis_len) - 1 ? body : ir::Block::Make({body}));
    }
    return body;
  } else {
    ir::Expr body = ir::Store::Make(tensor, tensor->body(), axis_exprs);
    // create schedule block itervars, i0,i1...
    std::vector<ir::Var> block_vars;
    for (int i = 0; i < shape.size(); ++i) {
      block_vars.push_back(Var(
          Expr(0), shape[i], cinn::UniqName("i" + std::to_string(i)), false));
      optim::ReplaceVarWithExpr(&body, axis[i], block_vars[i]);
    }
    body = ir::ScheduleBlockRealize::Make(
        axis_exprs,
        ir::ScheduleBlock::Make(block_vars, {}, {}, tensor->name, body));
    for (int i = static_cast<int>(axis_len) - 1; i >= 0; --i) {
      ir::Var loop_var = axis[i];
      ir::Expr loop_extent = shape[i];
      body = ir::For::Make(loop_var,
                           Expr(0),
                           loop_extent,
                           ir::ForType::Serial,
                           ir::DeviceAPI::Host,
                           ir::Block::Make({body}));
    }
    return body;
  }
}

}  // namespace ast_gen_ius
}  // namespace cinn
