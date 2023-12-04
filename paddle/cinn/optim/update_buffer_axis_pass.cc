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

#include "paddle/cinn/optim/update_buffer_axis_pass.h"

#include <unordered_map>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/ir/utils/ir_replace.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace optim {

bool ExprMathEqual(const Expr& expr1, const Expr& expr2) {
  ir::Expr cmp_expr = common::AutoSimplify(ir::Sub::Make(expr1, expr2));
  // This is ugry code since AutoSimplify is not powerful enough. Modify it
  // after we make auto simplify better
  ir::Expr simplied = common::AutoSimplify(cmp_expr);
  int count = 0;
  while (simplied != cmp_expr) {
    cmp_expr = simplied;
    simplied = common::AutoSimplify(cmp_expr);
    ++count;
    // Control dead loop
    if (count >= 5) {
      break;
    }
  }
  return simplied.is_constant() && simplied.get_constant() == 0;
}

/**
 * This is a template pass to update the buffer access when using
 * single axis of a mult-dim tensor. For example, if the tensor t
 * t.shape = [2, 3, 4] and the buffer access is t[12 * k]
 * it is same as t[k, 0, 0]. It is easy for human to understand
 * they are the same but not easy for compiler.
 *
 * This class check the buffer access are the same and update those
 * same buffer access with the same index expr.
 *
 * Note! this is a temporary solution. Our symbolic simplify is not
 * powerful to simplify the 12 * k / 4 % 3 and so on. So we only handle
 * the simplest case. We can modify our class when we can simplify the
 * 12 * k / 4 % 3 well.
 */
class AnalyzeSingleAxisOfMultDimTensor : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::Store* op, Expr* expr) override {
    ir::Store* store = expr->As<ir::Store>();
    ir::Tensor tensor = store->tensor.as_tensor_ref();
    AnalyzeSingleAxisAccess(store->indices, tensor);
    ir::IRMutator<>::Visit(op, expr);
  }

  // Analyze the buffer access inside load
  void Visit(const ir::Load* op, Expr* expr) override {
    ir::Load* load = expr->As<ir::Load>();
    ir::Tensor tensor = load->tensor.as_tensor_ref();
    AnalyzeSingleAxisAccess(load->indices, tensor);
    ir::IRMutator<>::Visit(op, expr);
  }

  void AnalyzeSingleAxisAccess(const std::vector<Expr>& indices,
                               const ir::Tensor& tensor) {
    if (!tensor->buffer.defined() ||
        tensor->buffer->memory_type == ir::MemoryType::Heap) {
      return;
    }
    CHECK(indices.size() > 0) << "Buffer access indices is empty";
    const std::string& buffer_name = tensor->buffer->name;
    const std::vector<ir::Expr>& shape = tensor->shape;

    ir::Expr index_expr;
    if (indices.size() == 1 && shape.size() > 1) {
      index_expr = indices[0];
    } else if (indices.size() == shape.size()) {
      ir::Expr mul = Expr(1);
      index_expr = indices.back();
      for (int i = static_cast<int>(indices.size()) - 2; i >= 0; --i) {
        mul = ir::Mul::Make(shape[i + 1], mul);
        ir::Expr cur = ir::Mul::Make(indices[i], mul);
        index_expr = ir::Add::Make(cur, index_expr);
      }
    }
    index_expr = common::AutoSimplify(index_expr);

    if (!buffer_name_to_same_single_axis.count(buffer_name)) {
      buffer_name_to_same_single_axis[buffer_name] = index_expr;
      return;
    } else {
      const ir::Expr& stored_index_expr =
          buffer_name_to_same_single_axis[buffer_name];
      if (!ExprMathEqual(index_expr, stored_index_expr)) {
        buffer_name_to_same_single_axis.erase(buffer_name);
      }
    }
  }

  std::unordered_map<std::string, ir::Expr> buffer_name_to_same_single_axis;
};

class AnalyzeBufferAxis : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  // Analyze the buffer access inside store
  void Visit(const ir::Store* op, Expr* expr) override {
    ir::Store* store = expr->As<ir::Store>();
    ir::Tensor tensor = store->tensor.as_tensor_ref();
    AnalyzeTensorAxis(store->indices, tensor);
    ir::IRMutator<>::Visit(op, expr);
  }

  // Analyze the buffer access inside load
  void Visit(const ir::Load* op, Expr* expr) override {
    ir::Load* load = expr->As<ir::Load>();
    ir::Tensor tensor = load->tensor.as_tensor_ref();
    AnalyzeTensorAxis(load->indices, tensor);
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::ScheduleBlockRealize* x, Expr* expr) override {
    const ir::ScheduleBlock* schedule_block =
        x->schedule_block.As<ir::ScheduleBlock>();
    const std::vector<ir::Var>& iter_vars = schedule_block->iter_vars;
    const std::vector<ir::Expr>& iter_values = x->iter_values;
    for (int i = 0; i < iter_vars.size(); ++i) {
      iter_var_to_bind_expr_[iter_vars[i]->name] = iter_values[i];
    }
    ir::IRMutator<>::Visit(x, expr);
  }

 private:
  void AnalyzeTensorAxis(const std::vector<Expr>& indices,
                         const ir::Tensor& tensor) {
    if (!tensor->buffer.defined() ||
        tensor->buffer->memory_type == ir::MemoryType::Heap) {
      return;
    }

    const std::string& buffer_name = tensor->buffer->name;
    if (!buffer_name_access_same_index_expr.count(buffer_name)) {
      for (int i = 0; i < indices.size(); ++i) {
        buffer_name_access_same_index_expr[buffer_name][i] =
            GetIndexBindExpr(indices[i]);
      }
      return;
    }

    std::map<int, ir::Expr>& index_expr =
        buffer_name_access_same_index_expr[buffer_name];
    for (int i = 0; i < indices.size(); ++i) {
      if (index_expr.count(i)) {
        if (!ExprMathEqual(index_expr[i], GetIndexBindExpr(indices[i]))) {
          index_expr.erase(i);
        }
      }
    }
    if (index_expr.empty()) {
      buffer_name_access_same_index_expr.erase(buffer_name);
    }
  }

  ir::Expr GetIndexBindExpr(ir::Expr index) {
    if (index.as_var() && iter_var_to_bind_expr_.count(index.as_var()->name)) {
      return iter_var_to_bind_expr_[index.as_var()->name];
    }
    return index;
  }

 public:
  // Stores the buffer names, and its indice where always using same Expr to
  // access For example:
  //   _A[i * 3][j] = ...
  //   ... = _A[k][j]
  // The buffer name _A will map to {1 : j}, where 1 is the indice
  // having same expr j.
  std::unordered_map<std::string, std::map<int, ir::Expr>>
      buffer_name_access_same_index_expr;

 private:
  std::unordered_map<std::string, ir::Expr> iter_var_to_bind_expr_;
};

class ReplaceSameAxisToZero : public ir::IRMutator<> {
 public:
  ReplaceSameAxisToZero(
      const std::unordered_map<std::string, std::map<int, ir::Expr>>&
          buffer_name_access_same_index_expr,
      const std::unordered_map<std::string, ir::Expr>&
          buffer_name_to_same_single_axis)
      : buffer_name_access_same_index_expr_(buffer_name_access_same_index_expr),
        buffer_name_to_same_single_axis_(buffer_name_to_same_single_axis) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  // Analyze the buffer access inside store
  void Visit(const ir::Store* op, Expr* expr) override {
    ir::Store* store = expr->As<ir::Store>();
    ir::Tensor tensor = store->tensor.as_tensor_ref();
    ReplaceIndices(tensor, &(store->indices));
    ir::IRMutator<>::Visit(op, expr);
  }

  // Analyze the buffer access inside load
  void Visit(const ir::Load* op, Expr* expr) override {
    ir::Load* load = expr->As<ir::Load>();
    ir::Tensor tensor = load->tensor.as_tensor_ref();
    ReplaceIndices(tensor, &(load->indices));
    ir::IRMutator<>::Visit(op, expr);
  }

 private:
  void ReplaceIndices(const ir::Tensor& tensor, std::vector<Expr>* indices) {
    if (!tensor->buffer.defined() ||
        tensor->buffer->memory_type == ir::MemoryType::Heap) {
      return;
    }
    const std::string& buffer_name = tensor->buffer->name;
    if (buffer_name_access_same_index_expr_.count(buffer_name)) {
      for (auto p : buffer_name_access_same_index_expr_.at(buffer_name)) {
        int r = p.first;
        // After optimization, some load indice may be removed, so we need this
        // conditioin
        if (indices->size() > r) {
          ir::ir_utils::IrReplace(
              &(indices->at(r)), indices->at(r), ir::Expr(0));
        }
      }
      return;
    }
    if (buffer_name_to_same_single_axis_.count(buffer_name)) {
      indices->clear();
      indices->push_back(ir::Expr(0));
      return;
    }
  }

  const std::unordered_map<std::string, std::map<int, ir::Expr>>&
      buffer_name_access_same_index_expr_;
  const std::unordered_map<std::string, ir::Expr>&
      buffer_name_to_same_single_axis_;
};

void UpdateBufferAxisPass(ir::Expr* expr) {
  VLOG(6) << "Before UpdateBufferAxisPass, Expr = \n" << *expr;
  AnalyzeBufferAxis buffer_axis_analyzer;
  buffer_axis_analyzer(expr);
  for (auto p : buffer_axis_analyzer.buffer_name_access_same_index_expr) {
    VLOG(6) << "Buffer name: " << p.first;
    for (auto q : p.second) {
      VLOG(6) << "  Index: " << q.first << " Expr: " << q.second;
    }
  }
  AnalyzeSingleAxisOfMultDimTensor singler_axis_analyzer;
  singler_axis_analyzer(expr);
  for (auto p : singler_axis_analyzer.buffer_name_to_same_single_axis) {
    VLOG(6) << "Single axis Buffer name: " << p.first;
    VLOG(6) << "Single Expr: " << p.second;
  }
  ReplaceSameAxisToZero replacer(
      buffer_axis_analyzer.buffer_name_access_same_index_expr,
      singler_axis_analyzer.buffer_name_to_same_single_axis);
  replacer(expr);
  VLOG(6) << "After UpdateBufferAxisPass, Expr = \n" << *expr;
}

}  // namespace optim
}  // namespace cinn
