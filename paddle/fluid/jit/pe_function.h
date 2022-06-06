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

#pragma once

#include <iostream>
#include <vector>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/executor_cache.h"
#include "paddle/fluid/jit/base_function.h"
#include "paddle/fluid/jit/ivalue.h"

namespace paddle {
namespace jit {

class PEFunction : public BaseFunction {
 public:
  PEFunction(const framework::ProgramDesc &program_desc,
             const std::vector<std::string> param_name_for_program,
             const std::map<std::string, IValue> &all_param)
      : BaseFunction(program_desc, param_name_for_program, all_param) {}

  ~PEFunction() {}

  std::vector<IValue> operator()(const std::vector<IValue> &args) {
    // bool is_test = true;
    std::string prog_string;
    std::hash<std::string> string_hash;
    program_desc_.Proto()->SerializePartialToString(&prog_string);
    int64_t program_id = static_cast<int64_t>(string_hash(prog_string));
    const framework::BlockDesc &global_block = program_desc_.Block(0);
    int64_t start_op_index = 0;
    int64_t end_op_index = static_cast<int64_t>(global_block.OpSize());

    ShareIntoScope(args);
    std::vector<std::string> input_var_names;
    for (auto &name : schema_.input_args) {
      input_var_names.emplace_back(name.Name());
    }
    std::vector<std::string> output_var_names;
    for (auto &name : schema_.output_args) {
      output_var_names.emplace_back(name.Name());
    }

    std::vector<std::string> dout_var_names;
    if (end_op_index > start_op_index) {
      // TODO(dev): support other devices
      auto cache_info = framework::GetExecutorInfoFromCache(
          program_desc_, phi::CPUPlace(), start_op_index, end_op_index,
          /*is_grad=*/false, program_id, &scope_);
      auto &parallel_executor = cache_info.first;
      auto &skip_eager_delete_vars =
          framework::ExecutorInfoCache::Instance().SkipEagerDeleteVars(
              program_id, false);
      if (cache_info.second /*is_new_created*/) {
        parallel_executor->SkipMemoryReuse(/*scope_idx=*/0, input_var_names);
        skip_eager_delete_vars.insert(skip_eager_delete_vars.end(),
                                      output_var_names.begin(),
                                      output_var_names.end());
        skip_eager_delete_vars.insert(skip_eager_delete_vars.end(),
                                      dout_var_names.begin(),
                                      dout_var_names.end());
        framework::details::ParseSafeEagerDeletionSkipVars(
            program_desc_, end_op_index, output_var_names,
            &skip_eager_delete_vars);
      }
      parallel_executor->RunWithoutFetch(skip_eager_delete_vars);
    }
    VLOG(6) << framework::GenScopeTreeDebugInfo(&scope_);
    std::vector<IValue> res;
    FetchOutput(&res);
    return res;
  }
};

}  // namespace jit
}  // namespace paddle
