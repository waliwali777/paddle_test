// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/jit/function_schema.h"

namespace paddle {
namespace jit {

Argument::Argument(const std::string& name, bool is_out)
    : name_(name), is_output_(is_out) {}

const std::string& Argument::Name() const { return name_; }

const std::vector<std::string> FunctionSchema::GetInputArgNames() const {
  std::vector<std::string> input_arg_names;
  for (auto& arg : input_args) {
    input_arg_names.emplace_back(arg.Name());
  }
  return input_arg_names;
}

const std::vector<std::string> FunctionSchema::GetOutputArgNames() const {
  std::vector<std::string> output_arg_names;
  for (auto& arg : output_args) {
    output_arg_names.emplace_back(arg.Name());
  }
  return output_arg_names;
}

void FunctionSchema::AddInputArg(const std::string& name) {
  input_args.emplace_back(name, false);
}

void FunctionSchema::AddOutputArg(const std::string& name) {
  output_args.emplace_back(name, true);
}

FunctionInfo::FunctionInfo(const std::string& func_name,
                           const std::vector<std::string>& param_names,
                           const framework::ProgramDesc& program_desc)
    : func_name_(func_name),
      param_names_(param_names),
      program_desc_(program_desc) {
  // Parse FunctionSchema
  for (auto& in_name : program_desc_.GetFeedTargetNames()) {
    schema_.AddInputArg(in_name);
  }
  for (auto& out_name : program_desc_.GetFetchTargetNames()) {
    schema_.AddOutputArg(out_name);
  }
  // remove feed fetch op
  RemoveFeedFetch(&program_desc_);
}

const std::string& FunctionInfo::GetFunctionName() const { return func_name_; }

const framework::ProgramDesc& FunctionInfo::GetProgramDesc() const {
  return program_desc_;
}

const std::vector<std::string>& FunctionInfo::GetParamNames() const {
  return param_names_;
}

const std::vector<std::string> FunctionInfo::GetInputArgNames() const {
  return schema_.GetInputArgNames();
}

const std::vector<std::string> FunctionInfo::GetOutputArgNames() const {
  return schema_.GetOutputArgNames();
}

}  // namespace jit
}  // namespace paddle
