// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/backends/compiler.h"

#include <fstream>

#include "paddle/cinn/backends/llvm/runtime_symbol_registry.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/hlir/framework/visualize_helper.h"
#ifdef CINN_WITH_CUDA
#include "paddle/cinn/backends/codegen_cuda_dev.h"
#include "paddle/cinn/backends/codegen_cuda_host.h"
#include "paddle/cinn/backends/codegen_cuda_util.h"
#include "paddle/cinn/backends/nvrtc/nvrtc_util.h"
#include "paddle/cinn/runtime/cuda/cuda_module.h"
#include "paddle/cinn/runtime/cuda/cuda_util.h"
#include "paddle/cinn/runtime/flags.h"
#endif

DECLARE_string(cinn_source_code_save_path);
DECLARE_string(cinn_dump_group_lowered_func);
DECLARE_string(cinn_dump_group_source_code);
DECLARE_string(cinn_dump_group_ptx);
DECLARE_string(cinn_dump_group_instruction);

namespace cinn {
namespace backends {
using ir::Module;

static constexpr int DebugLogMaxLen = 30000;

void DumpCompilationInfo::DumpLoweredFunc() {
  if (FLAGS_cinn_dump_group_lowered_func.empty()) {
    return;
  }
  for (int idx = 0; idx < info_.lowered_funcs.size(); idx++) {
    auto dump_path = utils::StringFormat(
        "%s/fusion_group_%d", FLAGS_cinn_dump_group_lowered_func.c_str(), idx);
    if (!hlir::framework::MakeDirectory(
            dump_path, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH)) {
      LOG(WARNING) << "Failed to make directory: \"" << dump_path
                   << "\", the lowered functions for this group will not dump.";
    } else {
      auto dump_file = utils::StringFormat(
          "%s/%s", dump_path.c_str(), "lowered_function.txt");
      VLOG(7) << "Dump lower functions to: " << dump_file;
      std::ofstream of(dump_file, std::ios_base::out);
      if (of.is_open()) {
        of << info_.lowered_funcs[idx].front();
      } else {
        LOG(WARNING) << "Failed to open file: " << dump_file
                     << ", please check your path.";
      }
      of.close();
    }
  }
}

void DumpCompilationInfo::DumpSourceCode() {
  if (FLAGS_cinn_dump_group_source_code.empty()) {
    return;
  }
  for (int idx = 0; idx < info_.source_codes.size(); idx++) {
    auto dump_path = utils::StringFormat(
        "%s/fusion_group_%d", FLAGS_cinn_dump_group_source_code.c_str(), idx);
    if (!hlir::framework::MakeDirectory(
            dump_path, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH)) {
      LOG(WARNING) << "Failed to make directory: \"" << dump_path
                   << "\", the source code for this group will not dump.";
    } else {
      auto dump_file =
          utils::StringFormat("%s/%s", dump_path.c_str(), "source_code.cu");
      VLOG(7) << "Dump source code to: " << dump_file;
      std::ofstream of(dump_file, std::ios_base::out);
      if (of.is_open()) {
        of << info_.source_codes[idx];
      } else {
        LOG(WARNING) << "Failed to open file: " << dump_file
                     << ", please check your path.";
      }
      of.close();
    }
  }
}

void DumpCompilationInfo::DumpPtxCode() {
  if (FLAGS_cinn_dump_group_ptx.empty()) {
    return;
  }
  for (int idx = 0; idx < info_.source_ptxs.size(); idx++) {
    auto dump_path = utils::StringFormat(
        "%s/fusion_group_%d", FLAGS_cinn_dump_group_ptx.c_str(), idx);
    if (!hlir::framework::MakeDirectory(
            dump_path, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH)) {
      LOG(WARNING) << "Failed to make directory: \"" << dump_path
                   << "\", the source ptx for this group will not dump.";
    } else {
      auto dump_file =
          utils::StringFormat("%s/%s", dump_path.c_str(), "source_ptx.ptx");
      VLOG(7) << "Dump source ptx to: " << dump_file;
      std::ofstream of(dump_file, std::ios_base::out);
      if (of.is_open()) {
        of << info_.source_ptxs[idx];
      } else {
        LOG(WARNING) << "Failed to open file: " << dump_file
                     << ", please check your path.";
      }
      of.close();
    }
  }
}

void DumpCompilationInfo::DumpInstruction() {
  if (FLAGS_cinn_dump_group_instruction.empty()) {
    return;
  }
  for (int idx = 0; idx < info_.instructions.size(); idx++) {
    auto dump_path = utils::StringFormat(
        "%s/fusion_group_%d", FLAGS_cinn_dump_group_instruction.c_str(), idx);
    if (!hlir::framework::MakeDirectory(
            dump_path, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH)) {
      LOG(WARNING) << "Failed to make directory: \"" << dump_path
                   << "\", the instruction for this group will not dump.";
    } else {
      auto dump_file =
          utils::StringFormat("%s/%s", dump_path.c_str(), "instruction.txt");
      VLOG(7) << "Dump instruction to: " << dump_file;
      std::ofstream of(dump_file, std::ios_base::out);
      if (of.is_open()) {
        of << info_.instructions[idx]->DumpInstruction();
      } else {
        LOG(WARNING) << "Failed to open file: " << dump_file
                     << ", please check your path.";
      }
      of.close();
    }
  }
}

SourceCodePrint::SourceCodePrint() {
  if (!FLAGS_cinn_source_code_save_path.empty()) {
    LOG(INFO)
        << "The CINN auto generated source code will writing into file: \""
        << FLAGS_cinn_source_code_save_path << "\"";
    of.open(FLAGS_cinn_source_code_save_path, std::ios_base::out);
  }
}

SourceCodePrint::~SourceCodePrint() {
  if (of.is_open()) {
    of.close();
  }
}

void SourceCodePrint::write(const std::string& source_code) {
  std::lock_guard<std::mutex> guard(mtx_);
  if (of.is_open()) {
    of << source_code << std::endl;
  } else if (!FLAGS_cinn_source_code_save_path.empty()) {
    LOG(WARNING) << "Failed to open \"" << FLAGS_cinn_source_code_save_path
                 << "\", source code will print.";
    if (source_code.size() > DebugLogMaxLen) {
      LOG(INFO) << "[CUDA] source code-0:\n"
                << source_code.substr(0, DebugLogMaxLen);
      for (int i = 1; i * DebugLogMaxLen < source_code.size(); ++i) {
        LOG(INFO) << "[CUDA] source code-" << i << ":\n"
                  << source_code.substr(DebugLogMaxLen * i, DebugLogMaxLen);
      }
    } else {
      LOG(INFO) << "[CUDA] source code:\n" << source_code;
    }
  }
}

void Compiler::Build(const Module& module, const std::string& code) {
  if (target_.arch == Target::Arch::NVGPU) {
    CompileCudaModule(module, code);
  } else if (target_.arch == Target::Arch::X86) {
    CompileX86Module(module);
  } else {
    CINN_NOT_IMPLEMENTED
  }
}

std::string Compiler::GetSourceCode(const ir::Module& module) {
  if (target_.arch == Target::Arch::NVGPU) {
#ifdef CINN_WITH_CUDA
    auto _host_module_device_module_ =
        SplitCudaAndHostModule(module);  // NOLINT
    auto& host_module = std::get<0>(_host_module_device_module_);
    auto& device_module = std::get<1>(_host_module_device_module_);
    CodeGenCUDA_Dev codegen(target_);
    auto source_code = codegen.Compile(device_module);
    return source_code;
#else
    CINN_NOT_IMPLEMENTED
#endif
  } else {
    CINN_NOT_IMPLEMENTED
  }
}

void Compiler::BuildDefault(const Module& module) {
  if (target_.arch == Target::Arch::NVGPU) {
    CompileCudaModule(module);
  } else if (target_.arch == Target::Arch::X86) {
    CompileX86Module(module);
  } else {
    CINN_NOT_IMPLEMENTED
  }
}

void Compiler::CompileCudaModule(const Module& module,
                                 const std::string& code) {
#ifdef CINN_WITH_CUDA
  auto _host_module_device_module_ = SplitCudaAndHostModule(module);  // NOLINT
  auto& host_module = std::get<0>(_host_module_device_module_);
  auto& device_module = std::get<1>(_host_module_device_module_);
  VLOG(3) << "[CUDA] host module:\n" << host_module;

  VLOG(3) << "[CUDA] device module:\n" << device_module;
  std::string source_code;
  if (code.empty()) {
    CodeGenCUDA_Dev codegen(target_);
    source_code = codegen.Compile(device_module);
  } else {
    source_code = code;
  }
  CHECK(!source_code.empty())
      << "Compile CUDA C code failed from device module:\n"
      << device_module;
  VLOG(3) << "[CUDA] C:\n" << source_code;
  SourceCodePrint::GetInstance()->write(source_code);
  using runtime::cuda::CUDAModule;

  nvrtc::Compiler compiler;
  auto ptx = compiler(source_code);
  CHECK(!ptx.empty()) << "Compile PTX failed from source code:\n"
                      << source_code;
  cuda_module_.reset(new CUDAModule(ptx,
                                    compiler.compile_to_cubin()
                                        ? CUDAModule::Kind::CUBIN
                                        : CUDAModule::Kind::PTX));

  RuntimeSymbols symbols;
  for (auto& fn : device_module.functions()) {
    std::string kernel_fn_name = fn->name;
    auto fn_kernel = cuda_module_->GetFunction(0, kernel_fn_name);
    CHECK(fn_kernel);

    symbols.RegisterVar(kernel_fn_name + "_ptr_",
                        reinterpret_cast<void*>(fn_kernel));
  }

  engine_ = ExecutionEngine::Create(ExecutionOptions(), std::move(symbols));
  engine_->Link<CodeGenCUDA_Host>(host_module);

#else
  CINN_NOT_IMPLEMENTED
#endif
}

void Compiler::CompileX86Module(const Module& module) {
  engine_->Link<CodeGenX86>(module);
}

void Compiler::ExportObject(const std::string& path) {
  engine_->ExportObject(path);
}

void* Compiler::Lookup(absl::string_view fn_name) {
  CHECK(engine_);
  if (engine_->Lookup(fn_name) != nullptr) {
    return engine_->Lookup(fn_name);
  }
  return nullptr;
}

}  // namespace backends
}  // namespace cinn
