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
#include "cinn/runtime/flags.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_compiler.h"
#include "paddle/fluid/operators/cinn/cinn_launch_context.h"

namespace paddle {
namespace operators {

constexpr char kX[] = "X";
constexpr char kOutputs[] = "Out";
constexpr char kCompilationKey[] = "compilation_key";

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

// Launch cinn to execute compiled executable program and wait done
void LaunchCinnExecution(const CinnCompiledObject& compiled_obj,
                         const CinnLaunchContext& context, void* stream);

// Set cinn FLAGS (such as FLAGS_cinn_cudnn_deterministic) with paddle's FLAGS.
void SetCinnRuntimeFlags();

template <typename DeviceContext>
void* GetStream(const framework::ExecutionContext& ctx) {
  return nullptr;
}

#ifdef PADDLE_WITH_CUDA
template <>
void* GetStream<platform::CUDADeviceContext>(
    const framework::ExecutionContext& ctx);
#endif

}  // namespace details

template <typename DeviceContext, typename T>
class CinnLaunchOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto& scope = ctx.scope();
    const auto& place = ctx.GetPlace();
    void* stream = details::GetStream<DeviceContext>(ctx);
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
        compilation_key, inputs_name2tensor, target, stream);
    details::DebugCinnCompiledResult(cinn_compiled_object);

    const auto& launch_context = cinn_compiled_object.launch_context;
    // Step 3. Prepare arguments needed for the compiled executable program.
    launch_context->UpdateCapturedEnv(scope, place);
    if (!launch_context->IsArgumentsInitialized()) {
      VLOG(4) << "CinnLaunchOp prepare arguments";

      // 3.1 Prepare input variables: tensors of input variables have
      //     been initialized before graph compiled, just check the
      //     equiality between tensors of paddle and cinn.
      for (const auto& var_name : input_variable_names) {
        if (!launch_context->IsVariableUsed(var_name)) {
          // some input variables don't need for cinn because they are
          // eliminated by optimized passes or some cinn operators use
          // less variables
          VLOG(4) << "Input variable(" << var_name << ") not used by cinn";
          continue;
        }

        launch_context->AssignExternalVariable(var_name);
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
        PADDLE_ENFORCE_EQ(
            launch_context->IsVariableUsed(var_name), true,
            platform::errors::InvalidArgument(
                "Output variable(%s) not used by cinn", var_name));

        launch_context->AssignExternalVariable(var_name);
      }

      // 3.3 Prepare internal or temporary variables: Create a temporary
      //     scope to keep internal variables within graph or temporary
      //     variables needed by the compiled runtime program in addition.
      //     Here we directly use the names from CinnScope as Paddle variable
      //     names, because they will not be used outside the graph
      //     and should be destructed after computation finished.
      auto internal_variable_names = launch_context->GetInternalVariableNames();
      for (const auto& var_name : internal_variable_names) {
        launch_context->AssignInternalVariable(var_name);
      }
    }

    // Step 4. Set CINN runtime FLAGS, such as FLAGS_cinn_cudnn_deterministic.
    details::SetCinnRuntimeFlags();

    // Step 5. Launch CINN to execute the compiled executable program
    VLOG(4) << "Run Cinn compiled executable program with stream: " << stream;
    details::LaunchCinnExecution(cinn_compiled_object, *launch_context, stream);
    VLOG(4) << "CinnLaunchOp launch execution done.";
  }
};

}  // namespace operators
}  // namespace paddle
