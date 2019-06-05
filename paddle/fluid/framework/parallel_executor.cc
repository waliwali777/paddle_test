/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/parallel_executor.h"
#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/details/async_ssa_graph_executor.h"
#include "paddle/fluid/framework/details/fast_threaded_ssa_graph_executor.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/details/parallel_ssa_graph_executor.h"
#include "paddle/fluid/framework/details/scope_buffered_ssa_graph_executor.h"
#include "paddle/fluid/framework/details/threaded_ssa_graph_executor.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/reference_count_pass_helper.h"
#include "paddle/fluid/platform/profiler.h"

#ifdef WITH_GPERFTOOLS
#include "gperftools/profiler.h"
#endif
DEFINE_string(pe_profile_fname, "",
              "Profiler filename for PE, which generated by gperftools."
              "Only valid when compiled `WITH_PRIFILER=ON`. Empty if disable.");
DEFINE_bool(enable_parallel_graph, false,
            "Force disable parallel graph execution mode if set false.");

namespace paddle {
namespace framework {

static std::once_flag gProfileOnce;
#ifdef WITH_GPERFTOOLS
static bool gProfileStarted = false;
#endif

class ParallelExecutorPrivate {
 public:
  explicit ParallelExecutorPrivate(const std::vector<platform::Place> &places)
      : places_(places) {
    if (!FLAGS_pe_profile_fname.empty()) {
      std::call_once(gProfileOnce, [] {
#ifdef WITH_GPERFTOOLS
        ProfilerStart(FLAGS_pe_profile_fname.c_str());
        gProfileStarted = true;
#else
        LOG(WARNING) << "Paddle is not compiled with gperftools. "
          "FLAGS_pe_profile_fname will be ignored";
#endif
      });
    }
  }

  ~ParallelExecutorPrivate() {
    if (own_local_scope_) {
      for (size_t i = 1; i < local_scopes_.size(); ++i) {
        // Skip the first scope, since it is the global scope.
        Scope *local_scope = local_scopes_[i];
        if (global_scope_->HasKid(local_scope)) {
          global_scope_->DeleteScope(local_scope);
        }
      }
    }
  }

  ir::Graph *PrepareGCAndRefCnts(ir::Graph *graph, size_t max_memory_size);

  inline bool HasGarbageCollectors() const { return !gcs_.empty(); }

  void ResetRuntimeReferenceCount(const std::vector<std::string> &fetch_tensors,
                                  const std::string &fetched_var_name) {
    for (size_t i = 0; i < runtime_ref_cnts_.size(); ++i) {
      for (auto &pair : global_ref_cnts_[i]) {
        runtime_ref_cnts_[i][pair.first] = pair.second;
      }

      for (auto &fetch_name : fetch_tensors) {
        runtime_ref_cnts_[i].erase(fetch_name);
      }
      runtime_ref_cnts_[i].erase(fetched_var_name);
    }
  }

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  void InitNCCLCtxs(framework::Scope *scope, const BuildStrategy &bst) {
    VLOG(1) << "nccl comm num:" << bst.nccl_comm_num_ << ", nranks:" << nranks_
            << ", num_trainers:" << bst.num_trainers_
            << ", trainer_id:" << bst.trainer_id_;

    if (bst.use_hierarchical_allreduce_) {
      VLOG(1) << ", use_hierarchical_allreduce:"
              << bst.use_hierarchical_allreduce_ << ", inter_trainers_num:"
              << bst.hierarchical_allreduce_inter_nranks_
              << ", exter_trainers_num:"
              << bst.hierarchical_allreduce_exter_nranks_;
    }

    std::vector<ncclUniqueId *> flat_nccl_ids;
    if (nranks_ == 1) {
      // FIXME(gongwb): need not to create ncclid when nranks==1
      nccl_ctxs_.InitFlatCtxs(places_, flat_nccl_ids, bst.num_trainers_,
                              bst.trainer_id_);
      return;
    }

    if (bst.enable_parallel_graph_) {
      VLOG(1) << "use only one ncclid in pg model";

      ncclUniqueId *nccl_id = nullptr;

      std::string var_name = platform::GetFlatNCCLVarName(0);
      auto nccl_id_var = scope->FindVar(var_name);
      if (nccl_id_var) {
        nccl_id = nccl_id_var->GetMutable<ncclUniqueId>();
      } else {
        nccl_id = new ncclUniqueId();
        PADDLE_ENFORCE(platform::dynload::ncclGetUniqueId(nccl_id));
      }

      flat_nccl_ids.push_back(nccl_id);

      nccl_ctxs_.InitFlatCtxs(places_, flat_nccl_ids, bst.num_trainers_,
                              bst.trainer_id_);
      VLOG(1) << "init bst nccl context complete!";
      return;
    }

    // num_trainers ==1 && places > 1
    if (bst.num_trainers_ == 1) {
      nccl_ctxs_.InitFlatCtxs(places_, flat_nccl_ids, bst.num_trainers_,
                              bst.trainer_id_);
      return;
    }

    for (int i = 0; i < static_cast<int>(bst.nccl_comm_num_); i++) {
      std::string var_name = platform::GetFlatNCCLVarName(i);
      auto nccl_id_var = scope->FindVar(var_name);
      PADDLE_ENFORCE(nccl_id_var, "can't find %s nccl_id_var", var_name);
      auto nccl_id = nccl_id_var->GetMutable<ncclUniqueId>();
      flat_nccl_ids.push_back(nccl_id);
    }

    nccl_ctxs_.InitFlatCtxs(places_, flat_nccl_ids, bst.num_trainers_,
                            bst.trainer_id_);

    if (bst.use_hierarchical_allreduce_) {
      std::string var_name = platform::GetHierarchicalInterNCCLVarName();
      auto nccl_id_var = scope->FindVar(var_name);
      auto inter_nccl_id = nccl_id_var->GetMutable<ncclUniqueId>();

      std::vector<ncclUniqueId *> exter_nccl_ids;
      for (int i = 0; i < static_cast<int>(bst.nccl_comm_num_); i++) {
        std::string var_name = platform::GetHierarchicalExterNCCLVarName(i);
        auto nccl_id_var = scope->FindVar(var_name);
        PADDLE_ENFORCE(nccl_id_var, "can't find %s nccl_id_var", var_name);
        auto nccl_id = nccl_id_var->GetMutable<ncclUniqueId>();
        exter_nccl_ids.push_back(nccl_id);
      }
      nccl_ctxs_.InitHierarchicalCtxs(places_, inter_nccl_id, exter_nccl_ids,
                                      bst.num_trainers_, bst.trainer_id_,
                                      bst.hierarchical_allreduce_inter_nranks_,
                                      bst.hierarchical_allreduce_exter_nranks_);
    }
  }
#endif

  BuildStrategy build_strategy_;
  std::vector<platform::Place> places_;
  std::vector<Scope *> local_scopes_;
  Scope *global_scope_;  // not owned
  std::unique_ptr<details::SSAGraphExecutor> executor_;

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  platform::MultiNCCLContextMap nccl_ctxs_;
#endif
  bool own_local_scope_;
  bool use_cuda_;
  bool use_all_reduce_;
  size_t nranks_;

  // global_ref_cnts_ is only initialized when ParallelExecutor constructs, and
  // then keeps unchanged
  // Before each iteration, runtime_ref_cnts_ is reset to global_ref_cnts_
  std::vector<ir::ReferenceCountMap> global_ref_cnts_;
  std::vector<ir::AtomicReferenceCountMap> runtime_ref_cnts_;
  ir::GarbageCollectorMap gcs_;
};

ir::Graph *ParallelExecutorPrivate::PrepareGCAndRefCnts(
    ir::Graph *graph, size_t max_memory_size) {
  for (size_t i = 0; i < places_.size(); ++i) {
    auto &place = places_[i];
    if (gcs_.count(place) > 0) {
      continue;
    }
    std::unique_ptr<GarbageCollector> gc;
#ifdef PADDLE_WITH_CUDA
    if (platform::is_gpu_place(place)) {
      if (IsFastEagerDeletionModeEnabled()) {
        gc.reset(new UnsafeFastGPUGarbageCollector(
            boost::get<platform::CUDAPlace>(place), max_memory_size));
      } else {
        gc.reset(new StreamGarbageCollector(
            boost::get<platform::CUDAPlace>(place), max_memory_size));
      }
      VLOG(10) << "Created " << i << "-th GarbageCollector at " << place;
    } else {
#endif
      if (platform::is_cpu_place(place)) {
        gc.reset(new CPUGarbageCollector(boost::get<platform::CPUPlace>(place),
                                         max_memory_size));
        VLOG(10) << "Created GarbageCollector at " << place;
      } else {
        PADDLE_THROW("Unsupported place for garbage collection");
      }
#ifdef PADDLE_WITH_CUDA
    }
#endif

    gcs_.emplace(place, std::move(gc));
  }

  if (!gcs_.empty()) {
    std::vector<ir::LastLiveOpsOfVars> last_live_ops_of_vars;

    auto ref_cnt_pass =
        ir::PassRegistry::Instance().Get("reference_count_pass");
    ref_cnt_pass->SetNotOwned(ir::kGlobalReferenceCount, &global_ref_cnts_);
    ref_cnt_pass->SetNotOwned(ir::kLastLiveOpsOfVars, &last_live_ops_of_vars);
    graph = ref_cnt_pass->Apply(graph);
    VLOG(10) << "ReferenceCountPass Applied";

    auto eager_deletion_pass =
        ir::PassRegistry::Instance().Get("eager_deletion_pass");
    eager_deletion_pass->SetNotOwned(ir::kRuntimeReferenceCount,
                                     &runtime_ref_cnts_);
    eager_deletion_pass->SetNotOwned(ir::kGarbageCollector, &gcs_);
    eager_deletion_pass->SetNotOwned(ir::kLastLiveOpsOfVars,
                                     &last_live_ops_of_vars);
    eager_deletion_pass->SetNotOwned(ir::kAllPlaces, &places_);
    graph = eager_deletion_pass->Apply(graph);
    VLOG(10) << "EagerDeletionPass Applied";
  }
  return graph;
}

std::vector<Scope *> &ParallelExecutor::GetLocalScopes() {
  return member_->local_scopes_;
}

void ParallelExecutor::DropLocalExeScopes() {
  auto executor = dynamic_cast<details::ScopeBufferedSSAGraphExecutor *>(
      member_->executor_.get());
  if (executor) {
    executor->DropLocalExeScopes();
  }
}

bool ParallelExecutor::NeedCreateLocalExeScope() {
  auto executor = dynamic_cast<details::ScopeBufferedSSAGraphExecutor *>(
      member_->executor_.get());
  return executor && executor->NeedCreateLocalExeScope();
}

ParallelExecutor::ParallelExecutor(const std::vector<platform::Place> &places,
                                   const std::vector<std::string> &bcast_vars,
                                   const std::string &loss_var_name,
                                   Scope *scope,
                                   const std::vector<Scope *> &local_scopes,
                                   const ExecutionStrategy &exec_strategy,
                                   const BuildStrategy &build_strategy,
                                   ir::Graph *graph)
    : member_(new ParallelExecutorPrivate(places)) {
  member_->global_scope_ = scope;
  member_->use_cuda_ = exec_strategy.use_cuda_;
  member_->build_strategy_ = build_strategy;
  member_->use_all_reduce_ =
      build_strategy.reduce_ == BuildStrategy::ReduceStrategy::kAllReduce;
  member_->nranks_ = build_strategy.num_trainers_ * places.size();
#if defined(PADDLE_WITH_CUDA) && defined(_WIN32)
  if (member_->use_cuda_) {
    PADDLE_ENFORCE(places.size() == 1, "Windows can support Single GPU only.");
  }
#endif
  if (!member_->use_all_reduce_) {
    PADDLE_ENFORCE(places.size() > 1,
                   "If you set build_strategy.reduce with 'Reduce',"
                   "the number of places must be greater than 1.");
  }

  // Step 1. Bcast the bcast_vars to devs.
  // Create local scopes
  if (local_scopes.empty()) {
    member_->own_local_scope_ = true;
    member_->local_scopes_.emplace_back(member_->global_scope_);
    for (size_t i = 1; i < member_->places_.size(); ++i) {
      member_->local_scopes_.emplace_back(&scope->NewScope());
    }
  } else {
    member_->own_local_scope_ = false;
    PADDLE_ENFORCE_EQ(member_->places_.size(), local_scopes.size());
    for (size_t i = 0; i < member_->places_.size(); ++i) {
      member_->local_scopes_.emplace_back(&local_scopes[i]->NewScope());
    }
  }

  std::vector<ir::Graph *> graphs;
  if (build_strategy.async_mode_) {
    PADDLE_ENFORCE(!member_->use_cuda_,
                   "gpu mode does not support async_mode_ now!");
    graphs.push_back(graph);
    for (size_t i = 1; i < places.size(); ++i) {
      auto *tmp_graph = new ir::Graph(graph->OriginProgram());
      async_graphs_.emplace_back(tmp_graph);
      graphs.push_back(tmp_graph);
    }
  }

  // FIXME(Yancey1989): parallel graph mode get better performance
  // in GPU allreduce distributed training. Need an elegant way to
  // choice the execution strategy.
  build_strategy.enable_parallel_graph_ =
      EnableParallelGraphExecution(*graph, exec_strategy, build_strategy);
  if (build_strategy.enable_parallel_graph_)
    VLOG(0) << "The Executor would execute the graph by ParallelGraph "
               "Execution which can get better performance,"
            << "you can force it off by env FLAGS_enable_parallel_graph=0";

  if (member_->use_cuda_) {
// Bcast Parameters to all GPUs
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    member_->InitNCCLCtxs(scope, build_strategy);

    // Initialize device context's nccl comm, will be used by normal
    // Operators like sync_batch_norm, and collective ops.
    // NOTE: more than one ParallelExecutor with same place, the nccl comm will
    // be rewrite and there will be some problem.
    // NOTE: NCCL group-calls and non-group-calls can not use the same
    // NCCL communicator, so for ParallelGraph and Multi-Process mode, re-use
    // same communicators.
    for (size_t dev_id = 0; dev_id < member_->places_.size(); ++dev_id) {
      platform::DeviceContextPool &pool =
          platform::DeviceContextPool::Instance();
      auto *dev_ctx = static_cast<platform::CUDADeviceContext *>(
          pool.Get(member_->places_[dev_id]));
      auto &nccl_ctx =
          member_->nccl_ctxs_.DefaultFlatCtx()->at(member_->places_[dev_id]);
      dev_ctx->set_nccl_comm(nccl_ctx.comm());
    }
#endif
  }
  // broadcast parameters from the 0th device to others:
  auto need_broadcast = [&]() -> bool {
    if (build_strategy.num_trainers_ > 1) {
      // 1. num_tariners would be grater than 1 for nccl distributed training.
      return true;
    } else if (member_->local_scopes_.size() != 1 && local_scopes.empty()) {
      // 2. Only one trainer process, but ParallelExecutor hold multiple
      // devices.
      return true;
    }
    return false;
  };

  if (need_broadcast()) {
    BCastParamsToDevices(bcast_vars, build_strategy.trainer_id_);
  }
  // Startup Program has been run. All local scopes has correct parameters.

  // Step 2. Convert main_program to SSA form and dependency graph. Also, insert
  // ncclOp
  std::vector<ir::Graph *> async_graphs(places.size());
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  if (build_strategy.async_mode_) {
    VLOG(3) << "use local async mode";
    graph = build_strategy.Apply(graph, {member_->places_[0]}, loss_var_name,
                                 {member_->local_scopes_[0]}, 1,
                                 member_->use_cuda_, &member_->nccl_ctxs_);
    for (size_t i = 1; i < member_->places_.size(); ++i) {
      graphs[i] =
          build_strategy.Apply(graphs[i], {member_->places_[i]}, loss_var_name,
                               {member_->local_scopes_[i]}, 1,
                               member_->use_cuda_, &member_->nccl_ctxs_);
      async_graphs[i] = graphs[i];
    }
  } else {
    graph = build_strategy.Apply(graph, member_->places_, loss_var_name,
                                 member_->local_scopes_, member_->nranks_,
                                 member_->use_cuda_, &member_->nccl_ctxs_);
  }
#else
  if (build_strategy.async_mode_) {
    VLOG(3) << "use local async mode";
    graph = build_strategy.Apply(graph, {member_->places_[0]}, loss_var_name,
                                 {member_->local_scopes_[0]}, 1,
                                 member_->use_cuda_);
    for (size_t i = 1; i < member_->places_.size(); ++i) {
      graphs[i] = build_strategy.Apply(
          graphs[i], {member_->places_[i]}, loss_var_name,
          {member_->local_scopes_[i]}, 1, member_->use_cuda_);
      async_graphs[i] = graphs[i];
    }
  } else {
    graph = build_strategy.Apply(graph, member_->places_, loss_var_name,
                                 member_->local_scopes_, member_->nranks_,
                                 member_->use_cuda_);
  }
#endif

  auto max_memory_size = GetEagerDeletionThreshold();
  VLOG(10) << "Eager Deletion Threshold "
           << static_cast<float>(max_memory_size) / (1 << 30);
  if (max_memory_size >= 0) {
    graph = member_->PrepareGCAndRefCnts(graph,
                                         static_cast<size_t>(max_memory_size));
  }

  async_graphs[0] = graph;

  // Step 3. Create vars in each scope. Passes may also create new vars.
  //         skip control vars and empty vars
  std::vector<details::VariableInfo> var_infos;
  for (auto &node : graph->Nodes()) {
    if (node->IsVar() && !node->IsCtrlVar() && node->Var()) {
      var_infos.emplace_back();
      var_infos.back().name_ = node->Var()->Name();
      var_infos.back().type_ = node->Var()->GetType();
      var_infos.back().persistable_ = node->Var()->Persistable();
    }
  }

  // If the loss_var_name is given, the number of graph should be only one.
  if (loss_var_name.size()) {
    size_t graph_num = ir::GraphNum(*graph);
    if (graph_num > 1) {
      LOG(WARNING)
          << "The number of graph should be only one, "
             "but the current graph has "
          << ir::GraphNum(*graph)
          << " sub_graphs. If you want to see the nodes of the "
             "sub_graphs, you should use 'FLAGS_print_sub_graph_dir' "
             "to specify the output dir. NOTES: if you not do training, "
             "please don't pass loss_var_name.";
    }
  }

  if (build_strategy.async_mode_) {
    VLOG(3) << "use AsyncSSAGraphExecutor";
    member_->executor_.reset(new details::AsyncSSAGraphExecutor(
        exec_strategy, member_->local_scopes_, member_->places_, async_graphs));
  } else if (build_strategy.enable_parallel_graph_) {
    VLOG(3) << "use ParallelSSAGraphExecutor";
#ifdef PADDLE_WITH_CUDA
    // TODO(Yancey1989): Remove passing in the main_program when
    // allreduce_seq_pass doesn't need it as the attr.
    member_->executor_.reset(new details::ParallelSSAGraphExecutor(
        exec_strategy, member_->local_scopes_, member_->places_, graph));
#else
    PADDLE_THROW(
        "Paddle should be compiled with CUDA for ParallelGraph Execution.");
#endif
  } else {
    if (exec_strategy.type_ == ExecutionStrategy::kDefault) {
      VLOG(3) << "use ThreadedSSAGraphExecutor";
      member_->executor_.reset(new details::ThreadedSSAGraphExecutor(
          exec_strategy, member_->local_scopes_, member_->places_, graph));
    } else {
      VLOG(3) << "use FastThreadedSSAGraphExecutor";
      member_->executor_.reset(new details::FastThreadedSSAGraphExecutor(
          exec_strategy, member_->local_scopes_, member_->places_, graph));
    }
  }

  VLOG(3) << "use ScopeBufferedSSAGraphExecutor";
  if (!build_strategy.async_mode_) {
    member_->executor_.reset(new details::ScopeBufferedSSAGraphExecutor(
        exec_strategy, member_->local_scopes_, std::move(var_infos),
        member_->places_, std::move(member_->executor_)));
  }
}

void ParallelExecutor::BCastParamsToDevices(
    const std::vector<std::string> &vars, int trainer_id) const {
  VLOG(3) << "BCastParamsToDevices";
  // the initializing bcast, all vars would be bcast from device(0).
  for (auto &var : vars) {
    framework::Variable *main_var = member_->local_scopes_[0]->FindVar(var);
    if (main_var == nullptr || !main_var->IsType<LoDTensor>()) {
      continue;
    }

    auto &main_tensor = main_var->Get<LoDTensor>();
    if (!main_tensor.IsInitialized()) {
      VLOG(3) << "one in var not inited, return!";
      continue;
    }
    auto &dims = main_tensor.dims();
    if (paddle::platform::is_gpu_place(main_tensor.place())) {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
      std::vector<void *> buffers;
      buffers.reserve(member_->places_.size());
      size_t numel = main_tensor.numel();
      ncclDataType_t data_type = platform::ToNCCLDataType(main_tensor.type());
      for (size_t i = 0; i < member_->places_.size(); ++i) {
        auto place = member_->places_[i];
        void *buffer;

        if (i == 0 && trainer_id == 0) {
          buffer = const_cast<void *>(main_tensor.data<void>());
        } else {
          auto local_scope = member_->local_scopes_[i];
          auto *t = local_scope->Var(var)->GetMutable<LoDTensor>();
          t->Resize(dims);
          buffer = t->mutable_data(place, main_tensor.type());
        }
        buffers.push_back(buffer);
      }

      PADDLE_ENFORCE_EQ(member_->places_.size(), buffers.size(),
                        "variables' buffer size to bcast NOT equal to places");
      {
        auto *nccl_ctxs = member_->nccl_ctxs_.DefaultFlatCtx();
        platform::NCCLGroupGuard guard;
        for (size_t i = 0; i < member_->places_.size(); ++i) {
          auto &nccl_ctx = nccl_ctxs->at(member_->places_[i]);
          platform::dynload::ncclBcast(buffers[i], numel, data_type, 0,
                                       nccl_ctx.comm_, nccl_ctx.stream());
        }
        nccl_ctxs->WaitAll();
      }
#endif
    } else {
      platform::CPUPlace cpu;
      for (size_t i = 1; i < member_->places_.size(); ++i) {
        auto local_scope = member_->local_scopes_[i];
        auto *t = local_scope->Var(var)->GetMutable<LoDTensor>();

        auto copy_memory = [&] {
          t->Resize(dims);
          t->mutable_data(cpu, main_tensor.type());
          paddle::framework::TensorCopy(main_tensor, cpu, t);
        };

        auto share_memory = [&] { t->ShareDataWith(main_tensor); };

        // FIXME(zcd): LR_DECAY_COUNTER should not be shared. This is a hot fix.
        if (member_->build_strategy_.async_mode_) {
          share_memory();
        } else if (member_->use_all_reduce_ || member_->use_cuda_ ||
                   var == "@LR_DECAY_COUNTER@") {
          copy_memory();
        } else {
          share_memory();
        }
      }
    }
  }
}

void ParallelExecutor::Run(const std::vector<std::string> &fetch_tensors,
                           const std::string &fetched_var_name) {
  VLOG(3) << "enter ParallelExecutor Run";
#ifdef WITH_GPERFTOOLS
  if (gProfileStarted) {
    ProfilerFlush();
  }
#endif

  platform::RecordBlock b(0);
  if (member_->HasGarbageCollectors()) {
    member_->ResetRuntimeReferenceCount(fetch_tensors, fetched_var_name);
  }

  VLOG(3) << "ParallelExecutor begin to run member_->executor_->Run";
  auto fetch_data = member_->executor_->Run(fetch_tensors);
  *member_->global_scope_->Var(fetched_var_name)->GetMutable<FeedFetchList>() =
      fetch_data;
}

void ParallelExecutor::FeedTensorsIntoLocalScopes(
    const std::vector<std::unordered_map<std::string, LoDTensor>> &tensors) {
  PADDLE_ENFORCE_EQ(member_->local_scopes_.size(), tensors.size());

  for (size_t i = 0; i < tensors.size(); ++i) {
    auto &map = tensors[i];
    auto *scope = member_->local_scopes_[i];
    for (auto &pair : map) {
      auto *trg = scope->Var(pair.first)->GetMutable<LoDTensor>();
      trg->ShareDataWith(pair.second);
      trg->set_lod(pair.second.lod());
    }
  }
}

void ParallelExecutor::FeedAndSplitTensorIntoLocalScopes(
    const std::unordered_map<std::string, LoDTensor> &tensors) {
  for (auto pair : tensors) {
    auto lod_tensors = pair.second.SplitLoDTensor(member_->places_);
    PADDLE_ENFORCE_EQ(
        member_->places_.size(), lod_tensors.size(),
        "The number of samples of current batch is less than the count of "
        "devices, currently, it is not allowed. (%d vs %d)",
        member_->places_.size(), lod_tensors.size());
    for (size_t j = 0; j < member_->places_.size(); ++j) {
      // TODO(panxy0718): Do I need to delete this var?
      auto t =
          member_->local_scopes_[j]->Var(pair.first)->GetMutable<LoDTensor>();
      t->ShareDataWith(lod_tensors[j]);
      t->set_lod(lod_tensors[j].lod());
    }
  }
}

ParallelExecutor::~ParallelExecutor() {
  for (auto &p : member_->places_) {
    platform::DeviceContextPool::Instance().Get(p)->Wait();
  }
  delete member_;
}

bool ParallelExecutor::EnableParallelGraphExecution(
    const ir::Graph &graph, const ExecutionStrategy &exec_strategy,
    const BuildStrategy &build_strategy) const {
  if (!FLAGS_enable_parallel_graph) {
    return false;
  }

  bool enable_parallel_graph = true;

  for (ir::Node *node : graph.Nodes()) {
    if (node->IsVar() && node->Var()) {
      // TODO(Yancey1989): support sparse update in ParallelGraph mode.
      if (node->Var()->GetType() == proto::VarType::SELECTED_ROWS) {
        enable_parallel_graph = false;
        break;
      }
    } else if (node->IsOp() && node->Op()) {
      // TODO(Yancey1989): support pserver mode
      if (node->Op()->Type() == "send" || node->Op()->Type() == "recv") {
        enable_parallel_graph = false;
        break;
      }
    }
  }

  if (!member_->use_all_reduce_ || !member_->use_cuda_) {
    if (build_strategy.enable_sequential_execution_ ||
        exec_strategy.type_ == ExecutionStrategy::ExecutorType::kExperimental) {
      enable_parallel_graph = false;
    }
  }

#ifdef WIN32
  VLOG(1) << "Windows has no support to parallel graph, enable_parallel_graph "
             "would be forced to false.";
  enable_parallel_graph = false;
#endif

  return enable_parallel_graph;
}

}  // namespace framework
}  // namespace paddle

USE_PASS(reference_count_pass);
USE_PASS(eager_deletion_pass);
