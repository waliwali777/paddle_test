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

#include "paddle/fluid/pir/dialect/distributed/ir/dist_op.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_attribute.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_type.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/ir_context.h"

namespace paddle {
namespace dialect {

const char* ShardTensorOp::attributes_name[1] = {"tensor_dist_attr"};

void ShardTensorOp::VerifySig() {
  VLOG(4)
      << "Start Verifying inputs, outputs and attributes for: ShardTensorOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    PADDLE_ENFORCE_EQ(
        input_size,
        1u,
        phi::errors::PreconditionNotMet(
            "The size %d of inputs must be equal to 1.", input_size));
    PADDLE_ENFORCE((*this)
                       ->operand_source(0)
                       .type()
                       .isa<paddle::dialect::DenseTensorType>(),
                   phi::errors::PreconditionNotMet(
                       "Type validation failed for the 0th input."));
  }
  VLOG(4) << "Verifying attributes:";
  {
    auto& attributes = this->attributes();
    PADDLE_ENFORCE(attributes.count("tensor_dist_attr") > 0 &&
                       attributes.at("tensor_dist_attr")
                           .isa<paddle::dialect::TensorDistAttribute>(),
                   phi::errors::PreconditionNotMet(
                       "Type of attribute: tensor_dist_attr is not right."));
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    PADDLE_ENFORCE_EQ(
        output_size,
        1u,
        phi::errors::PreconditionNotMet(
            "The size %d of outputs must be equal to 1.", output_size));
    PADDLE_ENFORCE(
        (*this)->result(0).type().isa<paddle::dialect::DistDenseTensorType>(),
        phi::errors::PreconditionNotMet(
            "Type validation failed for the 0th output."));
  }
  VLOG(4) << "End Verifying for: ShardTensorOp.";
}

void ShardTensorOp::Build(pir::Builder& builder,
                          pir::OperationArgument& argument,
                          pir::Value input,
                          pir::AttributeMap attributes) {
  VLOG(4) << "Start build ShardOp";
  paddle::dialect::DenseTensorType input_tensor_type;
  if (input.type().isa<paddle::dialect::DenseTensorType>()) {
    input_tensor_type =
        input.type().dyn_cast<paddle::dialect::DenseTensorType>();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorType"));
  }

  PADDLE_ENFORCE(attributes.find("tensor_dist_attr") != attributes.end(),
                 phi::errors::NotFound(
                     "'tensor_dist_attr' Attribute is expected for ShardOp"));
  paddle::dialect::TensorDistAttribute tensor_dist_attr =
      attributes.at("tensor_dist_attr")
          .dyn_cast<paddle::dialect::TensorDistAttribute>();

  VLOG(4) << "Builder construction inputs";
  argument.AddInput(input);

  VLOG(4) << "Builder construction attributes";
  pir::Attribute operand_dist_attr =
      TensorDistAttribute::get(pir::IrContext::Instance(),
                               tensor_dist_attr.process_mesh_attr(),
                               tensor_dist_attr.dims_mapping(),
                               tensor_dist_attr.partial_status());
  std::vector<pir::Attribute> operand_dist_attrs =
      std::vector<Attribute>{operand_dist_attr};

  pir::Attribute result_dist_attr =
      TensorDistAttribute::get(pir::IrContext::Instance(),
                               tensor_dist_attr.process_mesh_attr(),
                               tensor_dist_attr.dims_mapping(),
                               tensor_dist_attr.partial_status());
  std::vector<pir::Attribute> result_dist_attrs =
      std::vector<TensorDistAttribute>{result_dist_attr};
  pir::Attribnute op_dist_attr = OperationDistAttribute::Get(
      process_mesh, operand_dist_attrs, result_dist_attrs);
  argument.AddAttribute("op_dist_attr", op_dist_attr);

  VLOG(4) << "Builder construction outputs";
  auto dims_mapping = tensor_dist_attr.dims_mapping();
  auto global_dims = input_tensor_type.dims();
  PADDLE_ENFORCE_EQ(dims_mapping.size() == global_dims.size(),
                    phi::errors::PreconditionNotMet(
                        "dims_mapping size %d does not match input size %d",
                        dims_mapping,
                        global_dims));
  std::vector<int> local_shape(global_dims.size());
  for (int i = 0; i < global_dims.size(); ++i) {
    if (dims_mapping[i] == -1) {
      continue;
    }
    PADDLE_ENFORCE_EQ(
        global_dims[i] % dims_mapping[i] != 0,
        phi::errors::PreconditionNotMet(
            "global_dims size %d can't be evenly devided by dims_mapping %d",
            global_dims[i],
            dims_mapping[i]));
    local_shape = global_dims[i] / dims_mapping[i];
  }

  pir::Type out_dist_tensor_type =
      paddle::dialect::DistDenseTensorType::get(pir::IrContext::Instance(),
                                                input_tensor_type,
                                                tensor_dist_attr,
                                                phi::make_ddim(local_shape));
  argument.AddOutput(out_dist_tensor_type);
}

}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ShardTensorOp)
