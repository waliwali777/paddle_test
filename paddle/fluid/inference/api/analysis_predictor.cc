// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/api/analysis_predictor.h"
#include <glog/logging.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/profiler.h"

DECLARE_bool(profile);
DECLARE_int32(paddle_num_threads);

namespace paddle {

using contrib::AnalysisConfig;

namespace {
bool IsPersistable(const framework::VarDesc *var) {
  if (var->Persistable() &&
      var->GetType() != framework::proto::VarType::FEED_MINIBATCH &&
      var->GetType() != framework::proto::VarType::FETCH_LIST) {
    return true;
  }
  return false;
}
}  // namespace

bool AnalysisPredictor::Init(
    const std::shared_ptr<framework::Scope> &parent_scope,
    const std::shared_ptr<framework::ProgramDesc> &program) {
  VLOG(3) << "Predictor::init()";
#if !defined(_WIN32)
  if (FLAGS_profile) {
    LOG(WARNING) << "Profiler is actived, might affect the performance";
    LOG(INFO) << "You can turn off by set gflags '-profile false'";
    auto tracking_device = config_.use_gpu ? platform::ProfilerState::kAll
                                           : platform::ProfilerState::kCPU;
    platform::EnableProfiler(tracking_device);
  }
#endif

  // no matter with or without MKLDNN
  paddle::platform::SetNumThreads(FLAGS_paddle_num_threads);

  if (config_.use_gpu) {
    place_ = paddle::platform::CUDAPlace(config_.device);
    LOG(WARNING) << "ir optimize only supports CPU currently, enable_ir_optim "
                    "is turned false.";
    config_.enable_ir_optim = false;
  } else {
    place_ = paddle::platform::CPUPlace();
  }

  executor_.reset(new paddle::framework::NaiveExecutor(place_));

  if (program) {
    PADDLE_ENFORCE_NOT_NULL(
        parent_scope,
        "Both program and parent_scope should be set in Clone mode.");
    scope_ = parent_scope;
  } else {
    paddle::framework::InitDevices(false);
    scope_.reset(new paddle::framework::Scope());
  }

  if (!program) {
    if (!LoadProgramDesc()) return false;

    // Optimize the program, and load parameters and modify them in the
    // scope_.
    // This will change the scope_ address.
    if (config_.enable_ir_optim) {
      OptimizeInferenceProgram();
    } else {
      // Load parameters
      LoadParameters();
    }

    // Create local variables
    sub_scope_ = &(scope_->NewScope());

  } else {
    // If the program is passed from external, no need to optimize it, this
    // logic is used in the clone scenario.
    inference_program_ = program;
    sub_scope_ = &(scope_->NewScope());
  }

  executor_->Prepare(sub_scope_, *inference_program_, 0,
                     config_.use_feed_fetch_ops);

  PADDLE_ENFORCE_NOT_NULL(sub_scope_);

  // If the parent_scope is passed, we assert that the persistable variables
  // are already created, so just create the no persistable variables.

  // If not cloned, the parameters should be loaded OptimizeInferenceProgram.
  // So in both cases, just the local variables are needed to load, not the
  // parematers.
  executor_->CreateVariables(*inference_program_, 0, false, sub_scope_);

  // Get the feed_target_names and fetch_target_names
  PrepareFeedFetch();

  return true;
}

bool AnalysisPredictor::Run(const std::vector<PaddleTensor> &inputs,
                            std::vector<PaddleTensor> *output_data,
                            int batch_size) {
  VLOG(3) << "Predictor::predict";
  inference::Timer timer;
  timer.tic();
  // set feed variable
  std::vector<framework::LoDTensor> feeds;
  framework::Scope *scope = sub_scope_ ? sub_scope_ : scope_.get();
  if (!SetFeed(inputs, scope)) {
    LOG(ERROR) << "fail to set feed";
    return false;
  }

  // Run the inference program
  // if share variables, we need not create variables
  executor_->Run();

  // get fetch variable
  if (!GetFetch(output_data, scope)) {
    LOG(ERROR) << "fail to get fetches";
    return false;
  }
  VLOG(3) << "predict cost: " << timer.toc() << "ms";

  // Fix TensorArray reuse not cleaned bug.
  tensor_array_batch_cleaner_.CollectTensorArrays(scope_.get());
  tensor_array_batch_cleaner_.ResetTensorArray();
  return true;
}

bool AnalysisPredictor::SetFeed(const std::vector<PaddleTensor> &inputs,
                                framework::Scope *scope) {
  VLOG(3) << "Predictor::set_feed";
  if (inputs.size() != feeds_.size()) {
    LOG(ERROR) << "wrong feed input size, need " << feeds_.size() << " but get "
               << inputs.size();
    return false;
  }

  // Cache the inputs memory for better concurrency performance.
  feed_tensors_.resize(inputs.size());

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto &input = feed_tensors_[i];
    framework::DDim ddim = framework::make_ddim(inputs[i].shape);
    void *input_ptr;
    if (inputs[i].dtype == PaddleDType::INT64) {
      input_ptr = input.mutable_data<int64_t>(ddim, platform::CPUPlace());
    } else if (inputs[i].dtype == PaddleDType::FLOAT32) {
      input_ptr = input.mutable_data<float>(ddim, platform::CPUPlace());
    } else {
      LOG(ERROR) << "unsupported feed type " << inputs[i].dtype;
      return false;
    }

    // TODO(panyx0718): Init LoDTensor from existing memcpy to save a copy.
    std::memcpy(static_cast<void *>(input_ptr), inputs[i].data.data(),
                inputs[i].data.length());
    // TODO(Superjomn) Low performance, need optimization for heavy LoD copy.
    framework::LoD lod;
    for (auto &level : inputs[i].lod) {
      lod.emplace_back(level);
    }
    input.set_lod(lod);
    int idx = -1;
    if (config_.specify_input_name) {
      idx = feed_names_[inputs[i].name];
    } else {
      idx = boost::get<int>(feeds_[i]->GetAttr("col"));
    }
    framework::SetFeedVariable(scope, input, "feed", idx);
  }
  return true;
}

template <typename T>
void AnalysisPredictor::GetFetchOne(const framework::LoDTensor &fetch,
                                    PaddleTensor *output) {
  // set shape.
  auto shape = framework::vectorize(fetch.dims());
  output->shape.assign(shape.begin(), shape.end());
  // set data.
  const T *data = fetch.data<T>();
  int num_elems = inference::VecReduceToInt(shape);
  output->data.Resize(num_elems * sizeof(T));
  // The fetched tensor output by fetch op, should always in CPU memory, so just
  // copy.
  memcpy(output->data.data(), data, num_elems * sizeof(T));
  // set lod
  output->lod.clear();
  for (auto &level : fetch.lod()) {
    output->lod.emplace_back(level.begin(), level.end());
  }
}

bool AnalysisPredictor::GetFetch(std::vector<PaddleTensor> *outputs,
                                 framework::Scope *scope) {
  VLOG(3) << "Predictor::get_fetch";
  outputs->resize(fetchs_.size());
  for (size_t i = 0; i < fetchs_.size(); ++i) {
    int idx = boost::get<int>(fetchs_[i]->GetAttr("col"));
    PADDLE_ENFORCE((size_t)idx == i);
    framework::LoDTensor &fetch =
        framework::GetFetchVariable(*scope, "fetch", idx);
    auto type = fetch.type();
    auto output = &(outputs->at(i));
    if (type == typeid(float)) {
      GetFetchOne<float>(fetch, output);
      output->dtype = PaddleDType::FLOAT32;
    } else if (type == typeid(int64_t)) {
      GetFetchOne<int64_t>(fetch, output);
      output->dtype = PaddleDType::INT64;
    } else {
      LOG(ERROR) << "unknown type, only support float32 and int64 now.";
    }
  }
  return true;
}

void AnalysisPredictor::OptimizeInferenceProgram() {
  LOG(INFO) << "optimize begin";
  FLAGS_IA_enable_ir = config_.enable_ir_optim;
  FLAGS_IA_enable_tensorrt_subgraph_engine = false;
  FLAGS_IA_output_storage_path = "";  // Don't output the model.
  // Analyze inference_program
  if (!config_.model_dir.empty()) {
    argument_.fluid_model_dir.reset(new std::string(config_.model_dir));
  } else {
    PADDLE_ENFORCE(
        !config_.param_file.empty(),
        "Either model_dir or (param_file, prog_file) should be set.");
    PADDLE_ENFORCE(!config_.prog_file.empty());
    argument_.fluid_model_program_path.reset(
        new std::string(config_.prog_file));
    argument_.fluid_model_param_path.reset(new std::string(config_.param_file));
  }

  argument_.origin_program_desc.reset(
      new ProgramDesc(*inference_program_->Proto()));

  switch (config_.ir_mode) {
    case contrib::AnalysisConfig::IrPassMode::kExclude:
      Analyzer()
          .IncludeAllIrPasses()
          .SetUseMkldnn(config_._use_mkldnn)
          .DisableIrPasses(config_.ir_passes)
          .Run(&argument_);
      break;
    case contrib::AnalysisConfig::IrPassMode::kInclude:
      Analyzer()
          .SetUseMkldnn(config_._use_mkldnn)
          .IncludeIrPasses(config_.ir_passes)
          .Run(&argument_);
      break;
    default:
      LOG(ERROR) << "Only kExclude and kInclude modes are support yet.";
  }

  CHECK(argument_.transformed_program_desc);
  VLOG(5) << "to prepare executor";
  inference_program_.reset(
      new framework::ProgramDesc(*argument_.transformed_program_desc));
  if (argument_.Has(framework::ir::kParamScopeAttr)) {
    // Update scope.
    scope_.reset(
        argument_.Release<framework::Scope>(framework::ir::kParamScopeAttr));
  }
  LOG(INFO) << "== optimize end ==";
}

template <>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor<
    AnalysisConfig, PaddleEngineKind::kAnalysis>(const AnalysisConfig &config) {
  VLOG(3) << "create AnalysisConfig";
  if (config.use_gpu) {
    // 1. GPU memeroy
    PADDLE_ENFORCE_GT(
        config.fraction_of_gpu_memory, 0.f,
        "fraction_of_gpu_memory in the config should be set to range (0., 1.]");
    PADDLE_ENFORCE_GE(config.device, 0, "Invalid device id %d", config.device);
    std::vector<std::string> flags;
    if (config.fraction_of_gpu_memory >= 0.0f ||
        config.fraction_of_gpu_memory <= 0.95f) {
      flags.push_back("dummpy");
      std::string flag = "--fraction_of_gpu_memory_to_use=" +
                         std::to_string(config.fraction_of_gpu_memory);
      flags.push_back(flag);
      VLOG(3) << "set flag: " << flag;
      framework::InitGflags(flags);
    }
  }

  std::unique_ptr<PaddlePredictor> predictor(new AnalysisPredictor(config));
  if (!dynamic_cast<AnalysisPredictor *>(predictor.get())->Init(nullptr)) {
    return nullptr;
  }
  return predictor;
}

void AnalysisPredictor::PrepareFeedFetch() {
  PADDLE_ENFORCE_NOT_NULL(sub_scope_);
  CreateFeedFetchVar(sub_scope_);
  for (auto *op : inference_program_->Block(0).AllOps()) {
    if (op->Type() == "feed") {
      int idx = boost::get<int>(op->GetAttr("col"));
      if (feeds_.size() <= static_cast<size_t>(idx)) {
        feeds_.resize(idx + 1);
      }
      feeds_[idx] = op;
      feed_names_[op->Output("Out")[0]] = idx;
    } else if (op->Type() == "fetch") {
      int idx = boost::get<int>(op->GetAttr("col"));
      if (fetchs_.size() <= static_cast<size_t>(idx)) {
        fetchs_.resize(idx + 1);
      }
      fetchs_[idx] = op;
    }
  }
}

void AnalysisPredictor::CreateFeedFetchVar(framework::Scope *scope) {
  PADDLE_ENFORCE_NOT_NULL(scope);
  auto *var = scope->Var("feed");
  var->GetMutable<framework::FeedFetchList>();
  var = scope->Var("fetch");
  var->GetMutable<framework::FeedFetchList>();
}

std::unique_ptr<ZeroCopyTensor> AnalysisPredictor::GetInputTensor(
    const std::string &name) {
  PADDLE_ENFORCE(executor_->scope()->FindVar(name), "no name called %s", name);
  std::unique_ptr<ZeroCopyTensor> res(
      new ZeroCopyTensor(static_cast<void *>(executor_->scope())));
  res->input_or_output_ = true;
  res->SetName(name);
  return res;
}

std::unique_ptr<ZeroCopyTensor> AnalysisPredictor::GetOutputTensor(
    const std::string &name) {
  PADDLE_ENFORCE(executor_->scope()->FindVar(name), "no name called %s", name);
  std::unique_ptr<ZeroCopyTensor> res(
      new ZeroCopyTensor(static_cast<void *>(executor_->scope())));
  res->input_or_output_ = false;
  res->SetName(name);
  return res;
}

bool AnalysisPredictor::ZeroCopyRun() {
  executor_->Run();
  // Fix TensorArray reuse not cleaned bug.
  tensor_array_batch_cleaner_.CollectTensorArrays(scope_.get());
  tensor_array_batch_cleaner_.ResetTensorArray();
  return true;
}

bool AnalysisPredictor::LoadProgramDesc() {
  // Initialize the inference program
  std::string filename;
  if (!config_.model_dir.empty()) {
    filename = config_.model_dir + "/__model__";
  } else if (!config_.prog_file.empty() && !config_.param_file.empty()) {
    // All parameters are saved in a single file.
    // The file names should be consistent with that used
    // in Python API `fluid.io.save_inference_model`.
    filename = config_.prog_file;
  } else {
    LOG(ERROR) << string::Sprintf(
        "not valid model path '%s' or program path '%s'.", config_.model_dir,
        config_.param_file);
    return false;
  }

  std::string pb_content;
  // Read binary
  std::ifstream fin(filename, std::ios::in | std::ios::binary);
  PADDLE_ENFORCE(static_cast<bool>(fin), "Cannot open file %s", filename);
  fin.seekg(0, std::ios::end);

  pb_content.resize(fin.tellg());
  fin.seekg(0, std::ios::beg);
  fin.read(&(pb_content.at(0)), pb_content.size());
  fin.close();

  // Create ProgramDesc
  framework::proto::ProgramDesc proto;
  proto.ParseFromString(pb_content);
  inference_program_.reset(new framework::ProgramDesc(proto));
  return true;
}

bool AnalysisPredictor::LoadParameters() {
  PADDLE_ENFORCE_NOT_NULL(inference_program_.get(),
                          "The inference program should be loaded first.");
  const auto &global_block = inference_program_->MutableBlock(0);

  // create a temporary program to load parameters.

  std::unique_ptr<framework::ProgramDesc> load_program(
      new framework::ProgramDesc());
  framework::BlockDesc *load_block = load_program->MutableBlock(0);
  std::vector<std::string> params;

  for (auto *var : global_block->AllVars()) {
    if (IsPersistable(var)) {
      VLOG(3) << "persistable variable's name: " << var->Name();

      framework::VarDesc *new_var = load_block->Var(var->Name());
      new_var->SetShape(var->GetShape());
      new_var->SetDataType(var->GetDataType());
      new_var->SetType(var->GetType());
      new_var->SetLoDLevel(var->GetLoDLevel());
      new_var->SetPersistable(true);

      if (!config_.param_file.empty()) {
        params.push_back(new_var->Name());
      } else {
        // append_op
        framework::OpDesc *op = load_block->AppendOp();
        op->SetType("load");
        op->SetOutput("Out", {new_var->Name()});
        op->SetAttr("file_path", {config_.model_dir + "/" + new_var->Name()});
        op->CheckAttrs();
      }
    }
  }

  if (!config_.param_file.empty()) {
    // sort paramlist to have consistent ordering
    std::sort(params.begin(), params.end());
    // append just the load_combine op
    framework::OpDesc *op = load_block->AppendOp();
    op->SetType("load_combine");
    op->SetOutput("Out", params);
    op->SetAttr("file_path", {config_.param_file});
    op->CheckAttrs();
  }

  // Use NaiveExecutor to Load parameters.
  platform::CPUPlace place;
  framework::NaiveExecutor e(place);
  e.Prepare(scope_.get(), *load_program, 0, false);
  e.Run();

  return true;
}

AnalysisPredictor::~AnalysisPredictor() {
#if !defined(_WIN32)
  if (FLAGS_profile) {
    platform::DisableProfiler(platform::EventSortingKey::kTotal,
                              "./profile.log");
  }
#endif
  if (sub_scope_) {
    scope_->DeleteScope(sub_scope_);
  }
}

std::unique_ptr<PaddlePredictor> AnalysisPredictor::Clone() {
  auto *x = new AnalysisPredictor(config_);
  x->Init(scope_, inference_program_);
  return std::unique_ptr<PaddlePredictor>(x);
}

template <>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor<contrib::AnalysisConfig>(
    const contrib::AnalysisConfig &config) {
  return CreatePaddlePredictor<contrib::AnalysisConfig,
                               PaddleEngineKind::kAnalysis>(config);
}

}  // namespace paddle
