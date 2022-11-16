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
#include "paddle/fluid/framework/new_executor/standalone_executor.h"

#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"

DECLARE_bool(use_graph_engine);

namespace paddle {
namespace framework {
StandaloneExecutor::StandaloneExecutor(const platform::Place& place,
                                       const ProgramDesc& prog)
    : place_(place), prog_(prog) {}

paddle::framework::FetchList StandaloneExecutor::Run(
    Scope* scope,
    const std::vector<std::string>& feed_names,
    const std::vector<std::string>& fetch_names) {
  platform::RecordEvent record_event(
      "StandaloneExecutor::run", platform::TracerEventType::UserDefined, 1);
  if (FLAGS_use_graph_engine) {
    auto graph_engine_ = GetGraphEngine(scope, prog_, place_);
    VLOG(4) << "StandaloneExecutor: " << this
            << ", GraphEngine: " << graph_engine_.get() << " Run.";

    graph_engine_->SetGraph(prog_, feed_names, fetch_names, false);
    return graph_engine_->Run(feed_names, fetch_names);
  } else {
    auto core =
        GetInterpreterCore(scope, prog_, feed_names, fetch_names, false);
    VLOG(4) << "StandaloneExecutor: " << this << ", InterpreterCore: " << core;
    return core->Run(feed_names);
  }
}

framework::interpreter::CostInfo StandaloneExecutor::DryRun(
    Scope* scope,
    const std::vector<std::string>& feed_names,
    const std::vector<phi::DenseTensor>& feed_tensors) {
  if (FLAGS_use_graph_engine) {
    PADDLE_THROW(platform::errors::Unavailable(
        "DryRun is not supportted on GraphEngine."));
  } else {
    auto core = GetInterpreterCore(scope, prog_, feed_names, {}, true);
    return core->DryRun(feed_names, feed_tensors);
  }
}

std::shared_ptr<InterpreterCore> StandaloneExecutor::GetInterpreterCore(
    Scope* scope,
    const ProgramDesc& prog,
    const std::vector<std::string>& feed_names,
    const std::vector<std::string>& fetch_names,
    bool add_fetch_op) {
  std::ostringstream oss;
  oss << "feed:";
  for (auto& feedname : feed_names) {
    oss << feedname << ",";
  }
  oss << "fetch:";
  for (auto& fetchname : fetch_names) {
    oss << fetchname << ",";
  }
  oss << "scope:" << scope;

  auto iter = interpretercores_.find(oss.str());

  if (iter == interpretercores_.end()) {
    VLOG(3) << "create interpreter_core for " << oss.str() << " on place "
            << place_;
    VLOG(3) << "add fetch op: " << add_fetch_op;
    std::shared_ptr<InterpreterCore> core = nullptr;

    if (add_fetch_op) {
      core = CreateInterpreterCore(place_, prog, scope, fetch_names);
    } else {
      core = std::make_shared<InterpreterCore>(
          place_,
          prog.Block(0),
          /*skip_gc_vars=*/std::set<std::string>(),
          scope);
    }
    interpretercores_.emplace(oss.str(), core);
    return core;
  } else {
    return iter->second;
  }
}

std::shared_ptr<GraphEngine> StandaloneExecutor::GetGraphEngine(
    Scope* scope, const ProgramDesc& prog, const platform::Place& place) {
  std::ostringstream oss;
  oss << "scope:" << scope;

  auto iter = graph_engines_.find(oss.str());

  if (iter == graph_engines_.end()) {
    if (paddle::platform::is_custom_place(place)) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
      graph_engines_[oss.str()] =
          std::make_shared<CustomGraphEngine>(scope, prog, place);
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "Cannot use GraphEngine, Please recompile or reinstall paddlepaddle "
          "with CustomDevice."));
#endif
    } else {
      PADDLE_THROW(platform::errors::NotFound(
          "Not found GraphEngine of place %s", place));
    }
  }
  return graph_engines_[oss.str()];
}

}  // namespace framework
}  // namespace paddle
