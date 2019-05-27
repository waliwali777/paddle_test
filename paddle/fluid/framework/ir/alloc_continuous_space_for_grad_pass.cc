//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/alloc_continuous_space_for_grad_pass.h"
#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_registry.h"

DEFINE_uint64(fuse_parameter_memory_size, 0,  // 0 KB
              "fuse_parameter_memory_size is up limited memory size "
              "of one group parameters' gradient which is the input "
              "of communication calling(e.g NCCLAllReduce). "
              "The default value is 0, it means that "
              "not set group according to memory_size.");
DEFINE_int32(
    fuse_parameter_groups_size, 3,
    "fuse_parameter_groups_size is the size of one group parameters' gradient. "
    "The default value is a experimental result. If the "
    "fuse_parameter_groups_size is 1, it means that the groups size is "
    "the number of parameters' gradient. If the fuse_parameter_groups_size is "
    "-1, it means that there are only one group. The default value is 3, it is "
    "an experimental value.");

namespace paddle {
namespace framework {
namespace ir {
// SetFuseParameterGroupsSize and SetFuseParameterMemorySize are used in unit
// test, because it is invalid that seting 'FLAGS_fuse_parameter_memory_size'
// and 'FLAGS_fuse_parameter_groups_size' in unit test.
void SetFuseParameterGroupsSize(int group_size) {
  FLAGS_fuse_parameter_groups_size = group_size;
}

int GetFuseParameterGroupsSize() { return FLAGS_fuse_parameter_groups_size; }

void SetFuseParameterMemorySize(uint64_t memory_size) {
  FLAGS_fuse_parameter_memory_size = memory_size;
}

uint64_t GetFuseParameterMemorySize() {
  return FLAGS_fuse_parameter_memory_size;
}

static const char kUnKnow[] = "@UNKNOW@";

class AllocContinuousSpaceForGradPass : public ir::Pass {
 protected:
  void ApplyImpl(ir::Graph *graph) const {
    ir::Graph &result = *graph;

    auto &places = Get<const std::vector<platform::Place>>(details::kPlaces);
    auto &local_scopes = Get<const std::vector<Scope *>>(details::kLocalScopes);

    ResetAttribute<details::ParamsAndGrads>(details::kParamsAndGrads, &result);
    ResetAttribute<details::GroupParamsAndGrads>(details::kGroupParamsAndGrads,
                                                 &result);

    auto &params_grads =
        result.Get<details::ParamsAndGrads>(details::kParamsAndGrads);
    RecordParamsAndGrads(result, &params_grads);

    VLOG(10) << "The number of params and grads is:" << params_grads.size();
    if (params_grads.size() == 0) {
      return;
    }

    std::unordered_map<std::string, Node *> vars = GetVarInfo(result);
    auto &group_params_grads =
        result.Get<details::GroupParamsAndGrads>(details::kGroupParamsAndGrads);
    // Note: the order of params_grads may be changed by SetGroupParamsAndGrads.
    SetGroupParamsAndGrads(vars, params_grads, &group_params_grads);

    params_grads.clear();
    for (auto &group_p_g : group_params_grads) {
      params_grads.insert(params_grads.end(), group_p_g.begin(),
                          group_p_g.end());
    }

    if (IsUnifiedDtype(params_grads, vars)) {
      SetGradientPersistable(params_grads, vars);
      AllocContinuousAddressSpace(places, local_scopes, vars, params_grads,
                                  &result);
    } else {
      // Set Gradients as Persistable to prevent this var becoming reusable.
      for (auto &sub_param_grad : group_params_grads) {
        SetGradientPersistable(params_grads, vars);
        PADDLE_ENFORCE(IsUnifiedDtype(sub_param_grad, vars),
                       "The data type of the same group is not consistent.");
        AllocContinuousAddressSpace(places, local_scopes, vars, sub_param_grad,
                                    &result);
      }
    }
  }

  void SetGradientPersistable(
      const std::vector<std::pair<std::string, std::string>> &sub_param_grad,
      const std::unordered_map<std::string, Node *> &vars) const {
    for (auto &p_g : sub_param_grad) {
      // Get gradient var
      auto iter = vars.find(p_g.second);
      PADDLE_ENFORCE(iter != vars.end(), "%s is not found.", p_g.second);
      iter->second->Var()->SetPersistable(true);
      PADDLE_ENFORCE(IsSupportedVarType(iter->second->Var()->GetType()));
    }
  }

  bool IsUnifiedDtype(
      const details::ParamsAndGrads &params_grads,
      const std::unordered_map<std::string, Node *> &vars) const {
    auto dtype = this->GetDtypeOfVar(vars, params_grads.front().second);
    for (auto p_g : params_grads) {
      auto next_dtype = this->GetDtypeOfVar(vars, p_g.second);
      if (next_dtype != dtype) {
        return false;
      }
    }
    return true;
  }

  void AllocContinuousAddressSpace(
      const std::vector<platform::Place> &places,
      const std::vector<Scope *> &local_scopes,
      const std::unordered_map<std::string, Node *> &vars,
      const details::ParamsAndGrads &params_grads, Graph *result) const {
    // Create a FusedVarsSet to avoid duplicating names for fused_var in other
    // pass.
    if (!result->Has(details::kFusedVars)) {
      result->Set(details::kFusedVars, new details::FusedVars);
    }
    // the kFusedGrads is used be fuse_optimizer_op_pass.
    if (!result->Has(details::kFusedGrads)) {
      result->Set(details::kFusedGrads, new details::FusedGrads);
    }

    // the fused_var_name should be unique, so it appends
    // params_grads.begin()->second.
    auto fused_var_name = std::string(details::kFusedVarNamePrefix) + "@GRAD@" +
                          params_grads.begin()->second;
    result->Get<details::FusedGrads>(details::kFusedGrads)
        .emplace_back(fused_var_name);
    auto &fused_var_set = result->Get<details::FusedVars>(details::kFusedVars);
    PADDLE_ENFORCE_EQ(fused_var_set.count(fused_var_name), 0,
                      "%s is duplicate in FusedVars.", fused_var_name);
    fused_var_set.insert(fused_var_name);

    InitFusedVarsAndAllocSpaceForVars(places, local_scopes, vars,
                                      fused_var_name, params_grads);
  }

  std::unordered_map<std::string, Node *> GetVarInfo(
      const Graph &result) const {
    std::unordered_map<std::string, Node *> vars;
    for (Node *node : result.Nodes()) {
      if (node->IsVar() && node->Var()) {
        // Note: The graph may have the same name node. For example, parameter
        // is the input of operator and it also is the output of optimizer;
        vars.emplace(node->Var()->Name(), node);
      }
    }
    return vars;
  }

  template <typename AttrType>
  void ResetAttribute(const std::string &attr_name, ir::Graph *graph) const {
    if (graph->Has(attr_name)) {
      VLOG(10) << attr_name << " is reset.";
      graph->Erase(attr_name);
    }
    graph->Set(attr_name, new AttrType);
  }

  void SetGroupParamsAndGrads(
      const std::unordered_map<std::string, ir::Node *> &var_nodes,
      const details::ParamsAndGrads &params_grads,
      details::GroupParamsAndGrads *group_params_grads) const {
    SetGroupAccordingToLayers(var_nodes, params_grads, group_params_grads);
    SetGroupAccordingToMemorySize(var_nodes, group_params_grads);
    SetGroupAccordingToGroupSize(var_nodes, group_params_grads);
  }

  void SetGroupAccordingToLayers(
      const std::unordered_map<std::string, ir::Node *> &var_nodes,
      const details::ParamsAndGrads &params_grads,
      details::GroupParamsAndGrads *group_params_grads) const {
    using var_dtype = std::pair<std::string, proto::VarType::Type>;
    std::map<var_dtype, size_t> var_idx;

    for (size_t i = 0; i < params_grads.size(); ++i) {
      auto pos = params_grads[i].first.find_first_of(".");

      auto dtype = GetDtypeOfVar(var_nodes, params_grads[i].second);
      var_dtype var_key;
      if (pos == std::string::npos) {
        var_key = std::make_pair(std::string(kUnKnow), dtype);
      } else {
        var_key = std::make_pair(params_grads[i].first.substr(0, pos), dtype);
      }

      size_t idx = 0;
      auto var_idx_iter = var_idx.find(var_key);
      if (var_idx_iter != var_idx.end()) {
        idx = var_idx_iter->second;
      } else {
        group_params_grads->emplace_back();
        idx = group_params_grads->size() - 1;
        var_idx[var_key] = idx;
      }
      auto &local_group_params_grads = group_params_grads->at(idx);
      local_group_params_grads.emplace_back(
          std::make_pair(params_grads[i].first, params_grads[i].second));
    }

    if (VLOG_IS_ON(10)) {
      VLOG(10) << "SetGroupAccordingToLayers: ";
      for (size_t i = 0; i < group_params_grads->size(); ++i) {
        VLOG(10) << "group " << i;
        std::stringstream out;
        for (auto &p_g : group_params_grads->at(i)) {
          out << "(" << p_g.first << ", " << p_g.second << "), ";
        }
        VLOG(10) << out.str();
      }
    }
  }

  void SetGroupAccordingToMemorySize(
      const std::unordered_map<std::string, ir::Node *> &var_nodes,
      details::GroupParamsAndGrads *group_params_grads) const {
    const uint64_t group_memory_size = GetFuseParameterMemorySize();
    if (group_memory_size == 0) {
      return;
    }
    details::GroupParamsAndGrads local_group_params_grads;

    size_t j = 0;
    while (j < group_params_grads->size()) {
      local_group_params_grads.emplace_back();
      auto &group_p_g = local_group_params_grads.back();

      auto &grad_name = group_params_grads->at(j).front().second;
      auto var_type = GetDtypeOfVar(var_nodes, grad_name);

      size_t local_group_memory_size = 0;
      while (j < group_params_grads->size()) {
        std::for_each(
            group_params_grads->at(j).begin(), group_params_grads->at(j).end(),
            [&local_group_memory_size,
             &var_nodes](const std::pair<std::string, std::string> &p_g) {
              auto iter = var_nodes.find(p_g.second);
              PADDLE_ENFORCE(iter != var_nodes.end(), "%s is not found.",
                             p_g.second);

              size_t size =
                  framework::SizeOfType(iter->second->Var()->GetDataType());
              auto shape = iter->second->Var()->GetShape();
              std::for_each(shape.begin(), shape.end(),
                            [&size](const int64_t &n) { size *= n; });

              local_group_memory_size += size;
            });

        group_p_g.insert(group_p_g.end(), group_params_grads->at(j).begin(),
                         group_params_grads->at(j).end());

        ++j;
        if (local_group_memory_size >= group_memory_size ||
            j >= group_params_grads->size()) {
          break;
        }
        auto next_var_type =
            GetDtypeOfVar(var_nodes, group_params_grads->at(j).front().second);
        if (next_var_type != var_type) {
          break;
        }
      }
    }

    std::swap(*group_params_grads, local_group_params_grads);

    if (VLOG_IS_ON(10)) {
      VLOG(10) << string::Sprintf(
          "SetGroupAccordingToMemorySize(memory_size: %d):", group_memory_size);
      for (size_t i = 0; i < group_params_grads->size(); ++i) {
        VLOG(10) << "group " << i;
        std::stringstream out;
        for (auto &g_p : group_params_grads->at(i)) {
          auto iter = var_nodes.find(g_p.second);
          PADDLE_ENFORCE(iter != var_nodes.end(), "%s is not found.",
                         g_p.second);
          auto shape = iter->second->Var()->GetShape();
          size_t size =
              framework::SizeOfType(iter->second->Var()->GetDataType());
          std::for_each(shape.begin(), shape.end(),
                        [&size](const int64_t &n) { size *= n; });
          out << string::Sprintf("(%s(%d), %s)", g_p.second, size, g_p.first);
        }
        VLOG(10) << out.str();
      }
    }
  }

  void SetGroupAccordingToGroupSize(
      const std::unordered_map<std::string, ir::Node *> &var_nodes,
      details::GroupParamsAndGrads *group_params_grads) const {
    if (GetFuseParameterGroupsSize() == 1) {
      return;
    }
    const int group_size = GetFuseParameterGroupsSize() == -1
                               ? static_cast<int>(group_params_grads->size())
                               : GetFuseParameterGroupsSize();
    PADDLE_ENFORCE_GT(group_size, 1);
    size_t groups = (group_params_grads->size() + group_size - 1) / group_size;
    details::GroupParamsAndGrads local_group_params_grads;
    local_group_params_grads.reserve(groups);

    size_t j = 0;
    while (j < group_params_grads->size()) {
      local_group_params_grads.emplace_back();
      auto &group_p_g = local_group_params_grads.back();
      group_p_g.reserve(static_cast<size_t>(group_size));

      auto &grad_name = group_params_grads->at(j).front().second;
      auto var_type = GetDtypeOfVar(var_nodes, grad_name);

      while (j < group_params_grads->size()) {
        group_p_g.insert(group_p_g.end(), group_params_grads->at(j).begin(),
                         group_params_grads->at(j).end());
        ++j;
        if (j >= group_params_grads->size() || j % group_size == 0) {
          break;
        }
        auto next_var_type =
            GetDtypeOfVar(var_nodes, group_params_grads->at(j).front().second);
        if (next_var_type != var_type) break;
      }
    }
    std::swap(*group_params_grads, local_group_params_grads);

    if (VLOG_IS_ON(10)) {
      VLOG(10) << string::Sprintf(
          "SetGroupAccordingToGroupSize(group_size: %d):", group_size);
      for (size_t i = 0; i < group_params_grads->size(); ++i) {
        VLOG(10) << "group " << i;
        std::stringstream out;
        for (auto &p_g : group_params_grads->at(i)) {
          out << "(" << p_g.first << ", " << p_g.second << "), ";
        }
        VLOG(10) << out.str();
      }
    }
  }

  proto::VarType::Type GetDtypeOfVar(
      const std::unordered_map<std::string, Node *> &var_nodes,
      const std::string &name) const {
    auto grad_iter = var_nodes.find(name);
    PADDLE_ENFORCE(grad_iter != var_nodes.end());
    PADDLE_ENFORCE_NOT_NULL(grad_iter->second->Var());
    return grad_iter->second->Var()->GetDataType();
  }

 private:
  bool IsSupportedVarType(const proto::VarType::Type &type) const {
    // Current only support LOD_TENSOR.
    return type == proto::VarType::LOD_TENSOR;
  }

  void RecordParamsAndGrads(const ir::Graph &graph,
                            details::ParamsAndGrads *params_grads) const {
    std::vector<ir::Node *> topo_nodes = ir::TopologySortOperations(graph);
    for (auto &node : topo_nodes) {
      try {
        bool is_bk_op =
            static_cast<bool>(boost::get<int>(node->Op()->GetAttr(
                                  OpProtoAndCheckerMaker::OpRoleAttrName())) &
                              static_cast<int>(OpRole::kBackward));
        if (!is_bk_op) continue;
        // Currently, we assume that once gradient is generated, it can be
        // broadcast, and each gradient is only broadcast once.
        auto backward_vars =
            boost::get<std::vector<std::string>>(node->Op()->GetNullableAttr(
                OpProtoAndCheckerMaker::OpRoleVarAttrName()));
        PADDLE_ENFORCE_EQ(backward_vars.size() % 2, static_cast<size_t>(0));
        for (size_t i = 0; i < backward_vars.size(); i += 2) {
          VLOG(10) << "Trainable parameter: " << backward_vars[i]
                   << ", gradient: " << backward_vars[i + 1];

          params_grads->emplace_back(std::make_pair(
              backward_vars[i] /*param*/, backward_vars[i + 1] /*grad*/));
        }
      } catch (boost::bad_get e) {
      }
    }
  }

  void InitFusedVarsAndAllocSpaceForVars(
      const std::vector<platform::Place> &places,
      const std::vector<Scope *> &local_scopes,
      const std::unordered_map<std::string, ir::Node *> &vars,
      const std::string &fused_var_name,
      const details::ParamsAndGrads &params_grads) const {
    //  Init Gradients and FusedVars
    VLOG(10) << "Init FusedVars and Gradients.";
    for (auto it = local_scopes.rbegin(); it != local_scopes.rend(); ++it) {
      auto &scope = *it;

      PADDLE_ENFORCE(scope->FindVar(fused_var_name) == nullptr,
                     "%s has existed in scope.", fused_var_name);
      scope->Var(fused_var_name)->GetMutable<LoDTensor>();

      for (auto &p_g : params_grads) {
        auto iter = vars.find(p_g.second);
        PADDLE_ENFORCE(iter != vars.end());
        PADDLE_ENFORCE_NOT_NULL(iter->second->Var());
        PADDLE_ENFORCE_EQ(iter->second->Var()->GetType(),
                          proto::VarType::LOD_TENSOR);
        scope->Var(p_g.second)->GetMutable<LoDTensor>();
      }
    }

    // Alloc continuous space for vars.
    std::vector<std::string> grads_name;
    std::vector<std::string> params_name;
    grads_name.reserve(params_grads.size());
    params_name.reserve(params_grads.size());
    for (auto &p_g : params_grads) {
      params_name.emplace_back(p_g.first);
      grads_name.emplace_back(p_g.second);
    }
    framework::ProgramDesc program_desc;
    AppendAllocSpaceForVarsOp(params_name, grads_name, fused_var_name,
                              program_desc.MutableBlock(0));

    for (size_t i = 0; i < local_scopes.size(); ++i) {
      for (auto &op_desc : program_desc.Block(0).AllOps()) {
        auto op = OpRegistry::CreateOp(*op_desc);
        op->Run(*local_scopes[i], places[i]);
      }
    }
  }

  void AppendAllocSpaceForVarsOp(const std::vector<std::string> &params_name,
                                 const std::vector<std::string> &grads_name,
                                 const std::string &fused_var_name,
                                 BlockDesc *global_block) const {
    auto op_desc = global_block->AppendOp();
    op_desc->SetType("alloc_continuous_space");
    op_desc->SetInput("Input", params_name);
    op_desc->SetOutput("Output", grads_name);
    op_desc->SetOutput("FusedOutput", {fused_var_name});
  }
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(alloc_continuous_space_for_grad_pass,
              paddle::framework::ir::AllocContinuousSpaceForGradPass)
    .RequirePassAttr(paddle::framework::details::kPlaces)
    .RequirePassAttr(paddle::framework::details::kLocalScopes);
