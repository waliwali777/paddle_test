/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/paddle2cinn/cinn_graph_symbolization.h"

#include <algorithm>
#include <iterator>
#include <queue>
#include <vector>

#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/paddle2cinn/transform_desc.h"
#include "paddle/fluid/framework/variable.h"

#include "cinn/frontend/op_mappers/use_op_mappers.h"
#include "cinn/frontend/var_type_utils.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

using ir::Graph;
using ir::Node;
using CinnTensor = ::cinn::hlir::framework::Tensor;
using OpMapperContext = CinnGraphSymbolization::OpMapperContext;
using CinnOpDesc = CinnGraphSymbolization::CinnOpDesc;
using FeedInfoMap = CinnGraphSymbolization::FeedInfoMap;

namespace utils {

OpMapperContext::FeedInfo GetCinnFeedInfoFromTensor(const Tensor& tensor) {
  OpMapperContext::FeedInfo info;
  const auto& dim = tensor.dims();
  for (int i = 0; i < dim.size(); i++) {
    info.shape.emplace_back(static_cast<int>(dim[i]));
  }

  auto cinn_var_type = TransformVarTypeToCinn(tensor.type());
  info.type = ::cinn::frontend::utils::CppVarType2CommonType(cinn_var_type);
  return info;
}
}  // namespace utils

FeedInfoMap CinnGraphSymbolization::GetFeedInfoMapFromInput() const {
  FeedInfoMap feed_map;
  for (auto& feed_pair : input_tensors_) {
    const auto& feed_name = feed_pair.first;
    const auto* tensor = feed_pair.second;

    feed_map[feed_name] = utils::GetCinnFeedInfoFromTensor(*tensor);
  }
  return feed_map;
}

// get the graph's op input Parameter var name set
std::unordered_set<std::string>
CinnGraphSymbolization::GetGraphInputParameterNames() const {
  std::unordered_set<std::string> names;

  for (auto* node : graph_.Nodes()) {
    if (node->IsOp()) {
      for (auto* var : node->inputs) {
        if (var->Var()->IsParameter()) {
          // Only need preserve the input parameter var of graph,
          // others do not.
          names.insert(var->Name());
        }
      }
    }
  }

  return names;
}

// Transform paddle scope to cinn, note that we only preserve the graph’s
// input parameter variable and ignore others.
std::shared_ptr<::cinn::hlir::framework::Scope>
CinnGraphSymbolization::CreateCinnScope(const FeedInfoMap& feed_map) const {
  auto cinn_scope = ::cinn::hlir::framework::Scope::Create();

  // get the graph's input parameter variable name list
  auto parameter_names = GetGraphInputParameterNames();

  for (const auto& param_name : parameter_names) {
    // if cannot find var in graph input, skip.
    // scope accepte the CINN format name, so here we need transform
    // paddle format name to CINN format.
    auto* cinn_var = cinn_scope->Var<CinnTensor>(
        ::cinn::utils::TransValidVarName(param_name));

    auto& cinn_tensor = absl::get<CinnTensor>(*cinn_var);
    // here we only need preserve dtype and shape, do not need preserve data
    auto feed_info = feed_map.at(param_name);
    cinn_tensor->set_type(feed_info.type);
    cinn_tensor->Resize(::cinn::hlir::framework::Shape(feed_info.shape));
  }

  return cinn_scope;
}

std::vector<std::unique_ptr<CinnOpDesc>>
CinnGraphSymbolization::TransformAllGraphOpToCinn() const {
  std::vector<std::unique_ptr<CinnOpDesc>> cinn_op_descs;

  const auto& sorted_ops = ir::TopologySortOperations(graph_);
  for (auto* node : sorted_ops) {
    cinn_op_descs.emplace_back(std::make_unique<CinnOpDesc>());
    auto& cinn_desc = cinn_op_descs.back();

    TransformOpDescToCinn(node->Op(), cinn_desc.get());
  }
  return cinn_op_descs;
}

void CinnGraphSymbolization::RunOp(const CinnOpDesc& op_desc,
                                   const OpMapperContext& ctx) const {
  const auto& op_type = op_desc.Type();
  auto kernel = ::cinn::frontend::OpMapperRegistry::Global()->Find(op_type);
  PADDLE_ENFORCE_NE(
      kernel, nullptr,
      platform::errors::NotFound("Op %s Not Support by CINN", op_type.c_str()));
  VLOG(4) << "Running Op " << op_type;
  kernel->Run(op_desc, ctx);
}

void CinnGraphSymbolization::RunGraph(const OpMapperContext& ctx) const {
  auto cinn_op_descs = TransformAllGraphOpToCinn();
  // run the CINN op one by one, note that all ops
  // have been sorted at constructor.
  for (auto& op_desc : cinn_op_descs) {
    RunOp(*op_desc, ctx);
  }
}

::cinn::frontend::Program CinnGraphSymbolization::operator()() {
  std::string builder_name = "NetBuilder_of_graph_" + std::to_string(graph_id_);
  VLOG(4) << "NetBuilder Name " << builder_name;

  ::cinn::frontend::NetBuilder builder(builder_name);

  auto feed_map = GetFeedInfoMapFromInput();
  auto cinn_scope = CreateCinnScope(feed_map);

  OpMapperContext ctx(*cinn_scope, target_, &builder, &var_map_,
                      &var_model_to_program_map_);
  // add all tensor's feed info into context
  for (auto& feed_pair : feed_map) {
    ctx.AddFeedInfo(feed_pair.first, feed_pair.second);
  }
  RunGraph(ctx);

  return builder.Build();
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle
