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

#include "paddle/fluid/pir/dialect/distributed/transforms/dist_to_dense_pass.h"

#include <iostream>
#include <unordered_set>
#include <vector>

#include "paddle/common/flags.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_attribute.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_dialect.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_interface.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/attribute.h"

using paddle::dialect::DistDenseTensorType;

COMMON_DECLARE_bool(print_ir);

namespace paddle {
namespace dialect {

inline pir::Type CastToLocalType(pir::Type dist_type) {
  return dist_type.dyn_cast<DistTypeInterface>().local_type();
}

inline bool IsDistType(pir::Type type) {
  return type.dyn_cast<DistTypeInterface>() != nullptr;
}

void ProcessBlock(pir::Block* block) {
  for (auto iter = block->begin(); iter != block->end(); ++iter) {
    pir::Operation* op_item = &(*iter);
    VLOG(6) << "dist_to_dense main loop over op [" << op_item->name() << "].";

    for (size_t i = 0; i < op_item->num_results(); ++i) {
      auto result = op_item->result(i);
      auto origin_type = result.type();

      if IsDistType (origin_type) {
        auto local_type = CastToLocalType(origin_type);
        result.set_type(local_type);
      } else {
        // TODO(2024-Q2) not all value are dist type
        PADDLE_THROW(platform::errors::PreconditionNotMet(
            "The op [%s]'s [%d]th result is not Dist type.",
            op_item->name(),
            i));
      }
    }

    // TODO(2024-Q2) not all op are dist type
    auto& attributes = op_item->attributes();
    PADDLE_ENFORCE_EQ((attributes.HasAttribute(kAttrOpDistAttrs) &&
                       attributes.at(kAttrOpDistAttrs)
                           .isa<paddle::dialect::OperationDistAttribute>()),
                      true,
                      common::errors::PreconditionNotMet(
                          "The op [] has not op_dist_attr.", op_item->name()));
    attributes.erase(kAttrOpDistAttrs);

    // TODO(2024-Q2) Handle other special dist op in future.
  }
}

/* Verification:
    1. no operator has not OperatorDistAttr.
    2. all Values (Results) are DenseTensorType.
    3. no shard_tensor / reshard in block.
*/
void VerifyBlock(pir::Block* block) {
  for (auto iter = block->begin(); iter != block->end(); ++iter) {
    pir::Operation* op_item = &(*iter);

    for (size_t i = 0; i < op_item->num_results(); ++i) {
      auto result = op_item->result(i);

      PADDLE_ENFORCE_EQ(
          IsDistType(result.type()),
          true,
          phi::errors::PreconditionNotMet("Block still contain dist type."));
    }
    auto& attributes = op_item->attributes();
    PADDLE_ENFORCE_EQ(
        attributes.HasAttribute(kAttrOpDistAttrs),
        false,
        common::errors::PreconditionNotMet("The op [] still has op_dist_attr.",
                                           op_item->name()));
  }
}

std::shared_ptr<pir::Program> DistToDensePass(pir::Program* prog) {
  if (FLAGS_print_ir) {
    std::cout << "IR before DistToDense Pass = " << *prog << std::endl;
  }

  pir::IrMapping mapper;
  auto new_prog = prog->Clone(mapper);

  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<DistDialect>();

  ProcessBlock(new_prog->block());
  VerifyBlock(new_prog->block());

  if (FLAGS_print_ir) {
    std::cout << "IR after DistToDense Pass = " << *new_prog << std::endl;
  }

  return new_prog;
}

}  // namespace dialect
}  // namespace paddle
