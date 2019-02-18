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

#include "paddle/fluid/operators/benchmark/op_tester.h"
#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/pybind/pybind.h"

namespace paddle {
namespace operators {
namespace benchmark {

DEFINE_string(filename, "", "Path of op config file.");

static bool Has(const std::vector<std::string> &vec, const std::string &e) {
  for (auto &i : vec) {
    if (i == e) {
      return true;
    }
  }
  return false;
}

void OpTester::Init(const std::string &type, const VarShapeMap &input_shapes) {
  auto &op_desc_info = framework::OpInfoMap::Instance();
  // Initialize the OpDesc
  if (op_desc_info.Has(type)) {
    type_ = type;
    op_desc_.SetType(type);
    std::vector<std::string> input_names = GetOpProtoInputNames();
    for (auto &item : input_shapes) {
      std::string name = item.first;
      if (!Has(input_names, name) || item.second.size() < 1) {
        LOG(FATAL) << "The shape of input \"" << item.first
                   << "\" is not correctlly provided";
      }

      std::string var_name = type + "." + name;
      framework::VarDesc *var = Var(var_name);
      // Need to support more type
      var->SetType(framework::proto::VarType::LOD_TENSOR);
      var->SetPersistable(false);
      var->SetDataType(framework::proto::VarType::FP32);
      var->SetShape((item.second)[0]);

      op_desc_.SetInput(name, {var_name});
      inputs_.push_back(var_name);
    }
    std::vector<std::string> output_names = GetOpProtoOutputNames();
    for (auto &name : output_names) {
      std::string var_name = type + "." + name;
      framework::VarDesc *var = Var(var_name);
      // Need to support more type
      var->SetType(framework::proto::VarType::LOD_TENSOR);
      var->SetPersistable(false);
      var->SetDataType(framework::proto::VarType::FP32);

      op_desc_.SetOutput(name, {var_name});
      outputs_.push_back(var_name);
    }
  } else {
    LOG(FATAL) << "Op \"" << type << "\" is not registered.";
  }

  if (/* use_gpu */ false) {
    place_ = paddle::platform::CUDAPlace(0);
  } else {
    place_ = paddle::platform::CPUPlace();
  }

  framework::InitDevices(false);
  scope_.reset(new paddle::framework::Scope());

  op_ = framework::OpRegistry::CreateOp(op_desc_);
  CreateVariables(scope_.get());
}

void OpTester::Init(const std::string &filename) {
  config_ = OpTesterConfig(filename);

  auto &op_desc_info = framework::OpInfoMap::Instance();
  // Initialize the OpDesc
  if (op_desc_info.Has(config_.op_type)) {
    type_ = config_.op_type;
    op_desc_.SetType(config_.op_type);
    std::vector<std::string> input_names = GetOpProtoInputNames();
    for (auto &name : input_names) {
      const OpInputConfig *input = config_.GetInput(name);
      if (input == nullptr) {
        LOG(FATAL) << "Donot correctlly provide input " << name;
      }

      std::string var_name = config_.op_type + "." + name;
      framework::VarDesc *var = Var(var_name);
      // Need to support more type
      var->SetType(framework::proto::VarType::LOD_TENSOR);
      var->SetPersistable(false);
      var->SetDataType(framework::proto::VarType::FP32);
      var->SetShape(input->dims);

      op_desc_.SetInput(name, {var_name});
      inputs_.push_back(var_name);
    }
    std::vector<std::string> output_names = GetOpProtoOutputNames();
    for (auto &name : output_names) {
      std::string var_name = config_.op_type + "." + name;
      framework::VarDesc *var = Var(var_name);
      // Need to support more type
      var->SetType(framework::proto::VarType::LOD_TENSOR);
      var->SetPersistable(false);
      var->SetDataType(framework::proto::VarType::FP32);

      op_desc_.SetOutput(name, {var_name});
      outputs_.push_back(var_name);
    }
  } else {
    LOG(FATAL) << "Op \"" << config_.op_type << "\" is not registered.";
  }

  if (config_.use_gpu) {
    place_ = paddle::platform::CUDAPlace(0);
  } else {
    place_ = paddle::platform::CPUPlace();
  }

  framework::InitDevices(false);
  scope_.reset(new paddle::framework::Scope());

  op_ = framework::OpRegistry::CreateOp(op_desc_);
  CreateVariables(scope_.get());
}

void OpTester::Run() {
  if (config_.print_debug_string) {
    LOG(INFO) << DebugString();
  }

  // Warm up
  RunImpl();

  if (config_.profile) {
    if (platform::is_cpu_place(place_)) {
      platform::EnableProfiler(platform::ProfilerState::kCPU);
    } else {
#ifdef PADDLE_WITH_CUDA
      platform::EnableProfiler(platform::ProfilerState::kAll);
      // The default device_id of paddle::platform::CUDAPlace is 0.
      // Users can get the device_id using:
      //   int device_id = place.GetDeviceId();
      platform::SetDeviceId(0);
#else
      PADDLE_THROW("'CUDAPlace' is not supported in CPU only device.");
#endif
    }

    Timer timer;
    timer.tic();
    for (int i = config_.repeat; i > 0; --i) {
      RunImpl();
    }
    config_.runtime = timer.toc() / config_.repeat;
    platform::DisableProfiler(platform::EventSortingKey::kDefault,
                              "op_tester_profiler");
  } else {
    Timer timer;
    timer.tic();
    for (int i = config_.repeat; i > 0; --i) {
      RunImpl();
    }
    config_.runtime = timer.toc() / config_.repeat;
  }
  LOG(INFO) << "=== Run " << config_.repeat
            << " times, latency: " << config_.runtime << " ms ===";
}

void OpTester::RunImpl() {
  op_->Run(*scope_, place_);
  platform::DeviceContextPool::Instance().Get(place_)->Wait();
  scope_->DropKids();
}

std::vector<std::string> OpTester::GetOpProtoInputNames() {
  std::vector<std::string> input_names;
  const framework::proto::OpProto &proto =
      framework::OpInfoMap::Instance().Get(type_).Proto();
  for (int i = 0; i != proto.inputs_size(); ++i) {
    const auto &input = proto.inputs(i);
    input_names.push_back(input.name());
  }
  return input_names;
}

std::vector<std::string> OpTester::GetOpProtoOutputNames() {
  std::vector<std::string> output_names;
  const framework::proto::OpProto &proto =
      framework::OpInfoMap::Instance().Get(type_).Proto();
  for (int i = 0; i != proto.outputs_size(); ++i) {
    const auto &output = proto.outputs(i);
    output_names.push_back(output.name());
  }
  return output_names;
}

framework::VarDesc *OpTester::Var(const std::string &name) {
  auto it = vars_.find(name);
  if (it != vars_.end()) {
    return it->second.get();
  }
  auto *var = new framework::VarDesc(name);
  vars_[name].reset(var);
  return var;
}

template <typename T>
void OpTester::SetupTensor(framework::LoDTensor *tensor,
                           const std::vector<int64_t> &shape, T lower,
                           T upper) {
  static unsigned int seed = 100;
  std::mt19937 rng(seed++);
  std::uniform_real_distribution<double> uniform_dist(0, 1);

  T *ptr = tensor->mutable_data<T>(framework::make_ddim(shape), place_);
  if (platform::is_cpu_place(place_)) {
    for (int i = 0; i < tensor->numel(); ++i) {
      ptr[i] = static_cast<T>(uniform_dist(rng) * (upper - lower) + lower);
    }
  } else {
    framework::LoDTensor cpu_tensor;
    T *cpu_ptr = cpu_tensor.mutable_data<T>(framework::make_ddim(shape),
                                            platform::CPUPlace());
    for (int i = 0; i < cpu_tensor.numel(); ++i) {
      cpu_ptr[i] = static_cast<T>(uniform_dist(rng) * (upper - lower) + lower);
    }
    TensorCopySync(cpu_tensor, place_, tensor);
  }
}

void OpTester::CreateVariables(framework::Scope *scope) {
  for (auto &item : vars_) {
    auto &var = item.second;
    if (var->Name() == framework::kEmptyVarName) {
      continue;
    }

    auto *ptr = scope->Var(var->Name());
    framework::InitializeVariable(ptr, var->GetType());
    if (var->Persistable()) {
      VLOG(3) << "Create Variable " << var->Name()
              << " global, which pointer is " << ptr;
    } else {
      VLOG(3) << "Create Variable " << var->Name()
              << " locally, which pointer is " << ptr;
    }
  }

  // Allocate memory for input tensor
  for (auto &name : inputs_) {
    VLOG(3) << "Allocate memory for tensor " << name;
    auto &var_desc = vars_[name];
    std::vector<int64_t> shape = var_desc->GetShape();

    auto *var = scope->Var(name);
    auto *tensor = var->GetMutable<framework::LoDTensor>();
    SetupTensor<float>(tensor, shape, static_cast<float>(0.0),
                       static_cast<float>(1.0));
  }
}

static std::string GenSpaces(int count) {
  std::stringstream ss;
  for (int i = 0; i < count; ++i) {
    ss << "  ";
  }
  return ss.str();
}

std::string OpTester::DebugString() {
  std::stringstream ss;
  int count = 0;
  for (auto &item : vars_) {
    auto &var = item.second;
    ss << GenSpaces(count++) << "vars {\n";
    ss << GenSpaces(count) << "name: \"" << var->Name() << "\"\n";
    ss << GenSpaces(count++) << "type: {\n";
    ss << GenSpaces(count) << "type: LOD_TENSOR\n";
    ss << GenSpaces(count++) << "lod_tensor {\n";
    ss << GenSpaces(count++) << "tensor {\n";
    ss << GenSpaces(count) << "data_type: FP32\n";
    std::vector<int64_t> shape = var->GetShape();
    for (auto d : shape) {
      ss << GenSpaces(count) << "dims: " << d << "\n";
    }
    ss << GenSpaces(--count) << "}\n";
    ss << GenSpaces(--count) << "}\n";
    ss << GenSpaces(--count) << "}\n";
    ss << GenSpaces(count) << "persistable: " << var->Persistable() << "\n";
    ss << GenSpaces(--count) << "}\n";
  }
  ss << GenSpaces(count++) << "ops {\n";
  for (auto &name : op_desc_.InputNames()) {
    ss << GenSpaces(count++) << "inputs {\n";
    ss << GenSpaces(count) << "parameters: \"" << name << "\"\n";
    ss << GenSpaces(count) << "arguments: \"" << op_desc_.Input(name)[0]
       << "\"\n";
    ss << GenSpaces(--count) << "}\n";
  }
  for (auto &name : op_desc_.OutputNames()) {
    ss << GenSpaces(count++) << "outputs {\n";
    ss << GenSpaces(count) << "parameters: \"" << name << "\"\n";
    ss << GenSpaces(count) << "arguments: \"" << op_desc_.Output(name)[0]
       << "\"\n";
    ss << GenSpaces(--count) << "}\n";
  }
  ss << GenSpaces(count) << "type: " << op_desc_.Type() << "\n";
  ss << GenSpaces(--count) << "}\n";
  return ss.str();
}

TEST(op_tester, base) {
  OpTester tester;
  if (!FLAGS_filename.empty()) {
    tester.Init(FLAGS_filename);
  } else {
    VarShapeMap input_shapes;
    input_shapes["X"] = {{64, 64}};
    input_shapes["Y"] = {{64, 1}};
    tester.Init("elementwise_add", input_shapes);
  }
  tester.Run();
}

}  // namespace benchmark
}  // namespace operators
}  // namespace paddle
