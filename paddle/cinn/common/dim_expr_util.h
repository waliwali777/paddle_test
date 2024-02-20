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

#include <optional>
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace cinn::common {

symbol::DimExpr SubstituteDimExpr(
    const symbol::DimExpr& dim_expr,
    const std::unordered_map<symbol::DimExpr, symbol::DimExpr>&
        pattern_to_replacement);

}
