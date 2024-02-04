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

#pragma once

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/simplify_dim_expr_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/pir/core/utils.h"
#include "paddle/pir/dialect/shape/utils/dim_expr.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace {

template <typename DoEachT>
void VisitEachValue(const pir::Operation& op, const DoEachT& DoEach) {
  for (std::size_t i = 0; i < op.num_operands(); ++i) {
    DoEach(op.operand_source(i));
  }
  for (std::size_t i = 0; i < op.num_results(); ++i) {
    DoEach(op.result(i));
  }
}

void PrintOneValue(pir::Value value,
                   const pir::ShapeConstraintIRAnalysis& shape_analysis) {
  std::string str_shape_or_data = "shape: ";
  const symbol::ShapeOrDataDimExprs& ShapeOrData =
      shape_analysis.GetShapeOrDataForValue(value);
  for (const symbol::DimExpr& dim_expr : ShapeOrData.shape()) {
    str_shape_or_data.append(symbol::ToString(dim_expr)).append(" ");
  }
  str_shape_or_data.append("  data: ");
  if (ShapeOrData.data().has_value()) {
    for (const symbol::DimExpr& dim_expr : ShapeOrData.data().value()) {
      str_shape_or_data.append(symbol::ToString(dim_expr)).append(" ");
    }
  }
  VLOG(4) << "SimplifyDimExpr print :" << str_shape_or_data;
}

symbol::TensorShapeOrDataDimExprs SimplifyTensorShapeOrDataDimExprs(
    const symbol::TensorShapeOrDataDimExprs& tensor_shape_or_data) {
  std::vector<symbol::DimExpr> simplified_shape_dim_exprs;
  for (const symbol::DimExpr& shape_dim_expr : tensor_shape_or_data.shape()) {
    simplified_shape_dim_exprs.push_back(
        symbol::SimplifyDimExpr(shape_dim_expr));
  }
  if (!tensor_shape_or_data.data().has_value()) {
    symbol::ShapeOrData<symbol::DimExpr> simplified_shape_or_data(
        simplified_shape_dim_exprs);
    return simplified_shape_or_data;
  } else {
    std::vector<symbol::DimExpr> simplified_data_dim_exprs;
    for (const symbol::DimExpr& data_dim_expr :
         tensor_shape_or_data.data().value()) {
      simplified_data_dim_exprs.push_back(
          symbol::SimplifyDimExpr(data_dim_expr));
    }
    symbol::ShapeOrData<symbol::DimExpr> simplified_shape_or_data(
        simplified_shape_dim_exprs, simplified_data_dim_exprs);
    return simplified_shape_or_data;
  }
}

symbol::ShapeOrDataDimExprs SimplifyShapeOrData(
    const symbol::ShapeOrDataDimExprs& shape_or_data) {
  auto lambdas = symbol::Overloaded{
      [](const symbol::TensorShapeOrDataDimExprs& tensor_shape_or_data) {
        return symbol::ShapeOrDataDimExprs(
            SimplifyTensorShapeOrDataDimExprs(tensor_shape_or_data));
      },
      [](const symbol::TensorListShapeOrDataDimExprs&
             tensor_list_shape_or_data) {
        symbol::TensorListShapeOrDataDimExprs
            simplified_tensor_list_shape_or_data;
        for (symbol::TensorShapeOrDataDimExprs tensor_shape_or_data :
             tensor_list_shape_or_data) {
          simplified_tensor_list_shape_or_data.push_back(
              SimplifyTensorShapeOrDataDimExprs(tensor_shape_or_data));
        }
        return symbol::ShapeOrDataDimExprs(
            simplified_tensor_list_shape_or_data);
      }};
  return std::visit(lambdas, shape_or_data.variant());
}

void SimplifyOneValue(pir::Value value,
                      pir::ShapeConstraintIRAnalysis* shape_analysis) {
  if (!shape_analysis->HasShapeOrDataForValue(value)) {
    VLOG(4) << "SimplifyDimExpr shape_analysis.HasShapeOrDataForValue(value) "
               "return false";
    return;
  }
  const symbol::ShapeOrDataDimExprs& shape_or_data =
      shape_analysis->GetShapeOrDataForValue(value);
  symbol::ShapeOrDataDimExprs simplified_shape_or_data =
      SimplifyShapeOrData(shape_or_data);
  shape_analysis->SetShapeOrDataForValue(value, simplified_shape_or_data);
}

void SimplifyDimExpr(pir::ModuleOp module_op) {
  VLOG(4) << "SimplifyDimExpr start";
  pir::ShapeConstraintIRAnalysis shape_analysis =
      pir::ShapeAnalysisManager::Instance().Get(module_op.program());
  for (uint32_t i = 0; i < module_op->num_regions(); i++) {
    for (const pir::Block& block : module_op->region(i)) {
      for (const pir::Operation& op : block) {
        VLOG(1) << op.name();
        VisitEachValue(op, [&](pir::Value value) {
          PrintOneValue(value, shape_analysis);
          SimplifyOneValue(value, &shape_analysis);
          PrintOneValue(value, shape_analysis);
        });
      }
    }
    VLOG(4) << "SimplifyDimExpr end";
  }
}

class SimplifyDimExprPass : public pir::Pass {
 public:
  SimplifyDimExprPass() : pir::Pass("simplify_dim_expr_pass", 1) {}

  void Run(pir::Operation* op) override {
    pir::ModuleOp module_op = op->dyn_cast<pir::ModuleOp>();
    VLOG(4) << "SimplifyDimExprPass Run";
    SimplifyDimExpr(module_op);
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};

}  // namespace

std::unique_ptr<::pir::Pass> CreateSimplifyDimExprPass() {
  return std::make_unique<SimplifyDimExprPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
