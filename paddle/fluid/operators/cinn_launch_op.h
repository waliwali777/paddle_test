// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/scope.h"
#include "cinn/runtime/cinn_runtime.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_compiler.h"

namespace paddle {
namespace operators {

static constexpr char kX[] = "X";
static constexpr char kOutputs[] = "Out";
static constexpr char kCompilationKey[] = "compilation_key";

using LoDTensor = framework::LoDTensor;
using CinnTensor = ::cinn::hlir::framework::Tensor;
using CinnScope = ::cinn::hlir::framework::Scope;
using CinnCompiler = framework::paddle2cinn::CinnCompiler;
using CinnCompiledObject = framework::paddle2cinn::CinnCompiledObject;

namespace details {

// Tranform Paddle place to CINN target
const ::cinn::common::Target& PlaceToCinnTarget(const platform::Place& place);

// Print detailed compilation result of graph for debug
void DebugCinnCompiledResult(const CinnCompiledObject& result);

// Allocate buffer to a Paddle tensor with assginment information from CINN
void MutableTensorDataWithCompiledInfo(const platform::Place& place,
                                       const CinnTensor& cinn_tensor,
                                       LoDTensor* paddle_tensor);

class CinnLaunchContext {
 public:
  explicit CinnLaunchContext(const CinnCompiledObject& compiled_obj);

  // return whether a Paddle variable used on compiled kernels
  bool IsVariableUsed(const std::string& var_name);

  // Get CinnTensor with paddle variable name
  CinnTensor GetCinnTensor(const std::string& paddle_name);

  // Set an argument with name and tensor
  void SetArgument(const std::string& paddle_name, LoDTensor* paddle_tensor);

  // Extract temporary variable names from CinnScope by excluding
  // processed arguments in input and output variables
  std::vector<std::string> GetUnSetParameters();

  // Finalize all execution arguments and return them
  const std::map<std::string, cinn_pod_value_t>& FinalizeExecutionArguments();

 private:
  // Check whether tensors from Paddle and CINN respectively
  // of the same variable are equivalent in type and dimension
  void CheckTensorEquivalent(const std::string& paddle_name,
                             const LoDTensor* paddle_tensor,
                             const CinnTensor& cinn_tensor);

  // Share the buffer of a Paddle tensor to CINN by delivering memory address
  // to a cinn_buffer_t object
  std::unique_ptr<cinn_buffer_t> ShareTensorWithCinnBuffer(LoDTensor* tensor);

 private:
  const std::unordered_map<std::string, std::string>& paddle2cinn_varmap_;
  const std::shared_ptr<CinnScope> cinn_scope_;

  // all variables used in cinn compilation result
  std::unordered_set<std::string> cinn_variable_names_;

  // because a cinn_pod_value_t does not own the cinn_buffer_t object,
  // an extra stroage is necessary to keep the object and it can
  // not be released until runtime program finish  execution.
  std::vector<std::unique_ptr<cinn_buffer_t>> hold_buffers_;

  // name to execution argument
  std::map<std::string, cinn_pod_value_t> name2argument_;
};

}  // namespace details

template <typename DeviceContext, typename T>
class CinnLaunchOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto& scope = ctx.scope();
    const auto& place = ctx.GetPlace();
    // Step 1. Find graph object and prepare input
    PADDLE_ENFORCE_EQ(ctx.HasAttr(kCompilationKey), true,
                      platform::errors::NotFound(
                          "No Attribute(%s) found for CinnLaunchOp operator.",
                          kCompilationKey));
    const auto& compilation_key =
        ctx.template Attr<std::string>(kCompilationKey);
    VLOG(4) << "CinnLaunchOp attribute(" << kCompilationKey << ") "
            << "value:\n"
            << CinnCompiler::GetInstance()->ReadableKey(compilation_key);

    auto input_variable_names = ctx.InputNames(kX);
    const auto& input_tensors = ctx.MultiInput<LoDTensor>(kX);
    std::map<std::string, const LoDTensor*> inputs_name2tensor;
    std::transform(input_variable_names.begin(), input_variable_names.end(),
                   input_tensors.begin(),
                   std::inserter(inputs_name2tensor, inputs_name2tensor.end()),
                   [](const std::string& name, const LoDTensor* tensor) {
                     return std::make_pair(name, tensor);
                   });

    // Step 2. Get compilation result of the graph
    auto target = details::PlaceToCinnTarget(place);
    const auto& cinn_compiled_object = CinnCompiler::GetInstance()->Compile(
        compilation_key, inputs_name2tensor, target);
    details::DebugCinnCompiledResult(cinn_compiled_object);

    const auto& cinn_runtime_program = cinn_compiled_object.runtime_program;
    auto launch_context =
        std::make_unique<details::CinnLaunchContext>(cinn_compiled_object);

    // Step 3. Prepare arguments needed for the compiled executable program.
    VLOG(4) << "CinnLaunchOp prepare arguments";

    // 3.1 Prepare input variables: tensors of input variables have
    //     been initialized before graph compiled, just check the
    //     equiality between tensors of paddle and cinn.
    for (const auto& var_name : input_variable_names) {
      if (!launch_context->IsVariableUsed(var_name)) {
        // some input variables doesn't need for cinn because they are
        // eliminated by optimized passes or some cinn operators use
        // less variables
        VLOG(4) << "Input variable(" << var_name << ") not used by cinn";
        continue;
      }

      auto* tensor = scope.GetVar(var_name)->GetMutable<LoDTensor>();
      launch_context->SetArgument(var_name, tensor);
    }

    // 3.2 Prepare output variables: all output variables should
    //     be initialized and allocated buffer before
    //     the runtime program start execution, the compilation result
    //     includes details of their buffer assginment and we use that to
    //     allocate space in Paddle. For those variables allocated yet,
    //     like persistable parameters, just check the equiality between
    //     Paddle allocation and CINN buffer assginment.
    auto output_variable_names = ctx.OutputNames(kOutputs);
    for (const auto var_name : output_variable_names) {
      PADDLE_ENFORCE_EQ(launch_context->IsVariableUsed(var_name), true,
                        platform::errors::InvalidArgument(
                            "Output variable(%s) not used in cinn", var_name));

      auto* tensor = scope.GetVar(var_name)->GetMutable<LoDTensor>();
      if (!tensor->IsInitialized()) {
        details::MutableTensorDataWithCompiledInfo(
            place, launch_context->GetCinnTensor(var_name), tensor);
      }

      launch_context->SetArgument(var_name, tensor);
    }

    // 3.3 Prepare internal or temporary variables: Create a temporary
    //     scope to keep internal variables within graph or temporary
    //     variables needed by the compiled runtime program in addition.
    //     Here we directly use the names from CinnScope as Paddle variable
    //     names, because they will not be used outside the graph
    //     and should be destructed after computation finished.
    auto temp_variable_names = launch_context->GetUnSetParameters();
    auto temp_scope = scope.NewTmpScope();
    if (!temp_variable_names.empty()) {
      for (const auto& var_name : temp_variable_names) {
        auto* tensor = temp_scope->Var(var_name)->GetMutable<LoDTensor>();
        details::MutableTensorDataWithCompiledInfo(
            place, launch_context->GetCinnTensor(var_name), tensor);
        launch_context->SetArgument(var_name, tensor);
      }
    }

    // Step 4. Launch CINN to execute the compiled runtime program
    const auto& name2argument = launch_context->FinalizeExecutionArguments();
    cinn_runtime_program->Execute(&name2argument);
    VLOG(4) << "CinnLaunchOp launch execution done.";
  }
};

}  // namespace operators
}  // namespace paddle
