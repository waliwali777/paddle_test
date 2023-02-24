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

#include <sstream>
#include <string>
#include <tuple>

#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_analysis_config.h"
#include "paddle/fluid/inference/api/paddle_pass_builder.h"
#include "paddle/fluid/inference/utils/table_printer.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/phi/backends/cpu/cpu_info.h"
#include "paddle/utils/string/split.h"

#ifdef PADDLE_WITH_TENSORRT
#include "paddle/fluid/inference/tensorrt/helper.h"
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
DECLARE_uint64(initial_gpu_memory_in_mb);
#endif

namespace paddle {
struct MkldnnQuantizerConfig;

extern const std::vector<std::string> kTRTSubgraphPasses;
extern const std::vector<std::string> kDlnneSubgraphPasses;
extern const std::vector<std::string> kLiteSubgraphPasses;

AnalysisConfig::AnalysisConfig() {
  // NOTE(liuyuanle): Why put the following code here?
  // ref to https://github.com/PaddlePaddle/Paddle/pull/50864
  inference::InitGflagsFromEnv();
}

PassStrategy *AnalysisConfig::pass_builder() const {
  if (!pass_builder_.get()) {
    if (use_gpu_) {
      LOG(INFO) << "Create GPU IR passes";
      pass_builder_.reset(new GpuPassStrategy);
    } else if (use_xpu_) {
      pass_builder_.reset(new XpuPassStrategy);
    } else if (use_npu_) {
      pass_builder_.reset(new NpuPassStrategy);
    } else if (use_ipu_) {
      LOG(INFO) << "Create IPU IR passes";
      pass_builder_.reset(new IpuPassStrategy);
    } else {
      LOG(INFO) << "Create CPU IR passes";
      pass_builder_.reset(new CpuPassStrategy);
    }
  } else if (pass_builder_->use_gpu() ^ use_gpu()) {
    LOG(WARNING) << "The use_gpu flag is not compatible between Config and "
                    "PassBuilder, the flags are "
                 << use_gpu() << " " << pass_builder_->use_gpu();
    LOG(WARNING) << "Please make them compatible, still use the existing "
                    "PassBuilder.";
  }

  return pass_builder_.get();
}

AnalysisConfig::AnalysisConfig(const std::string &model_dir) {
  model_dir_ = model_dir;

  Update();
}
AnalysisConfig::AnalysisConfig(const std::string &prog_file,
                               const std::string &params_file) {
  prog_file_ = prog_file;
  params_file_ = params_file;

  Update();
}
void AnalysisConfig::SetModel(const std::string &prog_file_path,
                              const std::string &params_file_path) {
  prog_file_ = prog_file_path;
  params_file_ = params_file_path;

  Update();
}

void AnalysisConfig::EnableUseGpu(uint64_t memory_pool_init_size_mb,
                                  int device_id,
                                  Precision precision_mode) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  use_gpu_ = true;
  memory_pool_init_size_mb_ = memory_pool_init_size_mb;
  FLAGS_initial_gpu_memory_in_mb = memory_pool_init_size_mb_;
  gpu_device_id_ = device_id;
  mixed_precision_mode_ = precision_mode;
  if (precision_mode == Precision::kFloat32) {
    // default
  } else if (precision_mode == Precision::kHalf ||
             precision_mode == Precision::kBf16) {
    enable_gpu_mixed_ = true;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The Paddle-GPU inference currently only supports "
        "float32/float16/bfloat16 precision. Please check the parameters "
        "you specified in EnableUseGpu or enable_use_gpu function."));
  }
#else
  LOG(ERROR) << "Please use PaddlePaddle with GPU version.";
  use_gpu_ = false;
#endif

  Update();
}

void AnalysisConfig::Exp_EnableUseCutlass() {
#if defined(PADDLE_WITH_CUTLASS)
  use_cutlass_ = true;
#else
  LOG(ERROR) << "Please compile with cutlass to EnableUseCutlass()";
  use_cutlass_ = false;
#endif

  Update();
}

void AnalysisConfig::SetExecStream(void *stream) {
  PADDLE_ENFORCE_NOT_NULL(
      stream,
      platform::errors::InvalidArgument("`stream` should not be nullptr"));
  exec_stream_ = stream;
  use_external_stream_ = true;
  Update();
}

void *AnalysisConfig::GetExecStream() const {
  PADDLE_ENFORCE_NOT_NULL(
      exec_stream_,
      platform::errors::InvalidArgument("`stream` should not be nullptr"));
  return exec_stream_;
}

bool AnalysisConfig::external_stream_enabled() const {
  return use_external_stream_;
}

void AnalysisConfig::DisableGpu() {
  use_gpu_ = false;

  Update();
}

void AnalysisConfig::DisableFCPadding() {
  use_fc_padding_ = false;

  Update();
}

void AnalysisConfig::EnableXpu(int l3_workspace_size,
                               bool locked,
                               bool autotune,
                               const std::string &autotune_file,
                               const std::string &precision,
                               bool adaptive_seqlen,
                               bool enable_multi_stream) {
  use_xpu_ = true;
  xpu_l3_workspace_size_ = l3_workspace_size;
  xpu_locked_ = locked;
  xpu_autotune_ = autotune;
  xpu_autotune_file_ = autotune_file;
  xpu_precision_ = precision;
  xpu_adaptive_seqlen_ = adaptive_seqlen;
  xpu_enable_multi_stream_ = enable_multi_stream;
  Update();
}

void AnalysisConfig::SetXpuDeviceId(int device_id) {
  PADDLE_ENFORCE_EQ(use_xpu_,
                    true,
                    platform::errors::PreconditionNotMet(
                        "Should call EnableXpu before SetXpuDeviceId."));
  xpu_device_id_ = device_id;
  Update();
}

void AnalysisConfig::EnableNpu(int device_id) {
#if defined(PADDLE_WITH_ASCEND_CL)
  use_npu_ = true;
  npu_device_id_ = device_id;
#elif defined(PADDLE_WITH_CUSTOM_DEVICE)
  use_custom_device_ = true;
  custom_device_id_ = device_id;
  custom_device_type_ = "npu";
#else
  LOG(ERROR) << "Please compile with npu to EnableNpu()";
  use_npu_ = false;
#endif
  Update();
}

void AnalysisConfig::EnableCustomDevice(const std::string &device_type,
                                        int device_id,
                                        Precision precision_mode) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  use_custom_device_ = true;
  custom_device_id_ = device_id;
  custom_device_type_ = device_type;
  mixed_precision_mode_ = precision_mode;
  if (precision_mode == Precision::kFloat32) {
    // default
  } else if (precision_mode == Precision::kHalf ||
             precision_mode == Precision::kBf16) {
    enable_custom_device_mixed_ = true;
    LOG(INFO) << "enable_custom_device_mixed_";
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The Paddle-CustomDevice inference currently only supports "
        "float32/float16/bfloat16 precision. Please check the parameters "
        "you specified in EnableCustomDevice function."));
  }
#else
  LOG(ERROR) << "Please compile with CustomDevice to EnableCustomDevice()";
  use_custom_device_ = false;
#endif
  Update();
}

void AnalysisConfig::EnableIpu(int ipu_device_num,
                               int ipu_micro_batch_size,
                               bool ipu_enable_pipelining,
                               int ipu_batches_per_step) {
  enable_ir_optim_ = true;

  use_ipu_ = true;
  ipu_device_num_ = ipu_device_num;
  ipu_micro_batch_size_ = ipu_micro_batch_size;
  ipu_enable_pipelining_ = ipu_enable_pipelining;
  ipu_batches_per_step_ = ipu_batches_per_step;

  Update();
}

void AnalysisConfig::SetIpuConfig(bool ipu_enable_fp16,
                                  int ipu_replica_num,
                                  float ipu_available_memory_proportion,
                                  bool ipu_enable_half_partial,
                                  bool ipu_enable_model_runtime_executor) {
  ipu_enable_fp16_ = ipu_enable_fp16;
  ipu_replica_num_ = ipu_replica_num;
  ipu_available_memory_proportion_ = ipu_available_memory_proportion;
  ipu_enable_half_partial_ = ipu_enable_half_partial;
  ipu_enable_model_runtime_executor_ = ipu_enable_model_runtime_executor;

  Update();
}

void AnalysisConfig::SetIpuCustomInfo(
    const std::vector<std::vector<std::string>> &ipu_custom_ops_info,
    const std::map<std::string, bool> &ipu_custom_patterns) {
  ipu_custom_ops_info_ = ipu_custom_ops_info;
  for (auto iter = ipu_custom_patterns.begin();
       iter != ipu_custom_patterns.end();
       iter++) {
    if (iter->second == true) {
      ipu_custom_patterns_.push_back(
          std::vector<std::string>{iter->first, "True"});
    } else if (iter->second == false) {
      ipu_custom_patterns_.push_back(
          std::vector<std::string>{iter->first, "False"});
    }
  }

  Update();
}

void AnalysisConfig::LoadIpuConfig(const std::string &config_path) {
  std::ifstream fin(config_path, std::ios::in);
  PADDLE_ENFORCE_EQ(
      static_cast<bool>(fin.is_open()),
      true,
      platform::errors::NotFound(
          "Cannot open file %s, please confirm whether the file is normal.",
          config_path));
  std::string line;
  while (std::getline(fin, line)) {
    // remove all space
    line.erase(std::remove(line.begin(), line.end(), ' '), line.end());

    std::string key;
    std::string value;
    std::istringstream stream(line);
    // Split string to key and value based on the first `,`
    std::getline(stream, key, ',');
    std::getline(stream, value);

    auto string2bool = [](std::string s) {
      std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return ::tolower(c);
      });
      return s == "true" || s == "1";
    };

    // ipu_custom_ops_info:
    // [[paddle_op_name, popart_op_name, domain, version], [paddle_op_name,
    // popart_op_name, domain, version]...]
    // ipu_custom_patterns:
    // [[paddle_op_name, enable_pattern], [paddle_op_name, enable_pattern]...]
    auto string2vector = [](std::string s) {
      std::vector<std::vector<std::string>> custom_info;
      s.erase(0, 1);
      s.pop_back();

      std::string one;
      std::istringstream s_stream(s);
      while (std::getline(s_stream, one, ']')) {
        if (!one.empty()) {
          // remove `[`
          one.erase(0, 1);
          custom_info.push_back(paddle::string::Split(one, ','));
        }
      }
      return custom_info;
    };

    if (ipu_config_mapper_.find(key) == ipu_config_mapper_.end()) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "invalid key %s in IPU config: ", key));
    }
    switch (ipu_config_mapper_.at(key)) {
      case ipu_config_code::ipu_device_num:
        ipu_device_num_ = std::stoi(value);
        break;
      case ipu_config_code::ipu_micro_batch_size:
        ipu_micro_batch_size_ = std::stoi(value);
        break;
      case ipu_config_code::ipu_enable_pipelining:
        ipu_enable_pipelining_ = string2bool(value);
        break;
      case ipu_config_code::ipu_batches_per_step:
        ipu_batches_per_step_ = std::stoi(value);
        break;
      case ipu_config_code::ipu_enable_fp16:
        ipu_enable_fp16_ = string2bool(value);
        break;
      case ipu_config_code::ipu_replica_num:
        ipu_replica_num_ = std::stoi(value);
        break;
      case ipu_config_code::ipu_available_memory_proportion:
        ipu_available_memory_proportion_ = std::stof(value);
        break;
      case ipu_config_code::ipu_enable_half_partial:
        ipu_enable_half_partial_ = string2bool(value);
        break;
      case ipu_config_code::ipu_custom_ops_info:
        ipu_custom_ops_info_ = string2vector(value);
        break;
      case ipu_config_code::ipu_custom_patterns:
        ipu_custom_patterns_ = string2vector(value);
        break;
      case ipu_config_code::ipu_enable_model_runtime_executor:
        ipu_enable_model_runtime_executor_ = string2bool(value);
        break;
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "invalid key %s in IPU config", key));
        break;
    }
  }

  Update();
}

void AnalysisConfig::EnableONNXRuntime() {
#ifdef PADDLE_WITH_ONNXRUNTIME
  use_onnxruntime_ = true;
#else
  LOG(ERROR) << "Please compile with onnxruntime to EnableONNXRuntime()";
  use_onnxruntime_ = false;
#endif

  Update();
}

void AnalysisConfig::DisableONNXRuntime() {
  use_onnxruntime_ = false;
  Update();
}

void AnalysisConfig::EnableORTOptimization() {
#ifdef PADDLE_WITH_ONNXRUNTIME
  enable_ort_optimization_ = true;
#else
  LOG(ERROR) << "Please compile with onnxruntime to EnableORTOptimization()";
  enable_ort_optimization_ = false;
#endif

  Update();
}

AnalysisConfig::AnalysisConfig(const AnalysisConfig &other) {
#define CP_MEMBER(member__) member__ = other.member__;

  // Model related.
  CP_MEMBER(model_dir_);
  CP_MEMBER(model_from_memory_);  // the memory model reuses prog_file_ and
                                  // params_file_ fields.

  CP_MEMBER(opt_cache_dir_);
  CP_MEMBER(prog_file_);
  CP_MEMBER(params_file_);

  CP_MEMBER(use_fc_padding_);
  // GPU related.
  CP_MEMBER(use_gpu_);
  CP_MEMBER(use_cutlass_);
  CP_MEMBER(use_external_stream_);
  CP_MEMBER(exec_stream_);
  CP_MEMBER(use_cudnn_);
  CP_MEMBER(gpu_device_id_);
  CP_MEMBER(memory_pool_init_size_mb_);

  // Mixed precision related.
  CP_MEMBER(mixed_black_list_);
  CP_MEMBER(enable_gpu_mixed_);
  CP_MEMBER(mixed_precision_mode_);

  CP_MEMBER(enable_memory_optim_);
  // TensorRT related.
  CP_MEMBER(use_tensorrt_);
  CP_MEMBER(tensorrt_workspace_size_);
  CP_MEMBER(tensorrt_max_batchsize_);
  CP_MEMBER(tensorrt_min_subgraph_size_);
  CP_MEMBER(tensorrt_precision_mode_);
  CP_MEMBER(trt_disabled_ops_);
  CP_MEMBER(trt_use_dla_);
  CP_MEMBER(trt_dla_core_);
  CP_MEMBER(trt_use_static_engine_);
  CP_MEMBER(trt_use_calib_mode_);
  CP_MEMBER(trt_use_varseqlen_);
  CP_MEMBER(trt_with_interleaved_);
  CP_MEMBER(tensorrt_transformer_posid_);
  CP_MEMBER(tensorrt_transformer_maskid_);
  CP_MEMBER(trt_tuned_dynamic_shape_);
  CP_MEMBER(trt_allow_build_at_runtime_);
  CP_MEMBER(collect_shape_range_info_);
  CP_MEMBER(shape_range_info_path_);
  CP_MEMBER(trt_use_inspector_);
  CP_MEMBER(trt_engine_memory_sharing_);
  CP_MEMBER(trt_engine_memory_sharing_identifier_);
  // Dlnne related
  CP_MEMBER(use_dlnne_);
  CP_MEMBER(dlnne_min_subgraph_size_);
  CP_MEMBER(dlnne_max_batchsize_);
  CP_MEMBER(dlnne_use_static_batch_);
  CP_MEMBER(dlnne_weight_share_mode_);
  CP_MEMBER(dlnne_use_calib_mode_);
  CP_MEMBER(dlnne_precision_mode_);
  CP_MEMBER(dlnne_disable_nodes_by_outputs_);
  CP_MEMBER(dlnne_input_shape_dict_);
  // MKLDNN related.
  CP_MEMBER(use_mkldnn_);
  CP_MEMBER(mkldnn_enabled_op_types_);
  CP_MEMBER(mkldnn_cache_capacity_);
  // Bfloat16 related.
  CP_MEMBER(use_mkldnn_bfloat16_);
  CP_MEMBER(bfloat16_enabled_op_types_);
  // Quantization related.
  CP_MEMBER(use_mkldnn_int8_);
  CP_MEMBER(quantize_enabled_op_types_);
  CP_MEMBER(quantize_excluded_op_ids_);
  CP_MEMBER(use_mkldnn_quantizer_);
  CP_MEMBER(mkldnn_quantizer_config_);
  CP_MEMBER(min_input_shape_);
  CP_MEMBER(max_input_shape_);
  CP_MEMBER(optim_input_shape_);
  CP_MEMBER(disable_trt_plugin_fp16_);

  CP_MEMBER(use_lite_);
  CP_MEMBER(lite_precision_mode_);
  CP_MEMBER(lite_passes_filter_);
  CP_MEMBER(lite_ops_filter_);
  CP_MEMBER(lite_zero_copy_);

  // XPU related.
  CP_MEMBER(use_xpu_);
  CP_MEMBER(xpu_device_id_);
  CP_MEMBER(xpu_l3_workspace_size_);
  CP_MEMBER(xpu_locked_);
  CP_MEMBER(xpu_autotune_);
  CP_MEMBER(xpu_autotune_file_);
  CP_MEMBER(xpu_precision_);
  CP_MEMBER(xpu_adaptive_seqlen_);
  CP_MEMBER(xpu_enable_multi_stream_);

  // Lite OpenCL Related
  CP_MEMBER(use_opencl_);

  // NPU related.
  CP_MEMBER(use_npu_);
  CP_MEMBER(npu_device_id_);
  CP_MEMBER(nnadapter_config_);

  // profile related.
  CP_MEMBER(with_profile_);

  // cinn compiler related.
  CP_MEMBER(use_cinn_compiler_);

  // glog related.
  CP_MEMBER(with_glog_info_);

  // Ir related.
  CP_MEMBER(enable_ir_optim_);
  CP_MEMBER(use_feed_fetch_ops_);
  CP_MEMBER(ir_debug_);
  CP_MEMBER(specify_input_name_);

  CP_MEMBER(cpu_math_library_num_threads_);

  CP_MEMBER(serialized_info_cache_);

  CP_MEMBER(thread_local_stream_);

  // ipu related
  CP_MEMBER(use_ipu_);
  CP_MEMBER(ipu_device_num_);
  CP_MEMBER(ipu_micro_batch_size_);
  CP_MEMBER(ipu_enable_pipelining_);
  CP_MEMBER(ipu_batches_per_step_);
  CP_MEMBER(ipu_enable_fp16_);
  CP_MEMBER(ipu_replica_num_);
  CP_MEMBER(ipu_available_memory_proportion_);
  CP_MEMBER(ipu_enable_half_partial_);
  CP_MEMBER(ipu_enable_model_runtime_executor_);
  CP_MEMBER(ipu_custom_ops_info_);
  CP_MEMBER(ipu_custom_patterns_);

  // fleet exe related
  CP_MEMBER(dist_config_);

  // custom device related.
  CP_MEMBER(use_custom_device_);
  CP_MEMBER(custom_device_type_);
  CP_MEMBER(custom_device_id_);
  CP_MEMBER(enable_custom_device_mixed_);

  // JITLayer relate
  CP_MEMBER(apply_optim_);
  CP_MEMBER(skip_load_params_);

  if (use_gpu_) {
    PADDLE_ENFORCE_EQ(use_xpu_,
                      false,
                      platform::errors::InvalidArgument(
                          "Only one choice can be made between CPU and XPU."));
    pass_builder_.reset(new GpuPassStrategy(
        *static_cast<GpuPassStrategy *>(other.pass_builder())));
  } else if (use_ipu_) {
    pass_builder_.reset(new IpuPassStrategy(
        *static_cast<IpuPassStrategy *>(other.pass_builder())));
  } else if (use_xpu_) {
    pass_builder_.reset(new XpuPassStrategy(
        *static_cast<XpuPassStrategy *>(other.pass_builder())));
  } else if (use_npu_) {
    pass_builder_.reset(new NpuPassStrategy(
        *static_cast<NpuPassStrategy *>(other.pass_builder())));
  } else {
    pass_builder_.reset(new CpuPassStrategy(
        *static_cast<CpuPassStrategy *>(other.pass_builder())));
  }

#undef CP_MEMBER

  Update();
  if (use_tensorrt_ || use_cinn_compiler_) {
    // Update() will reset all the passes, when some tensorRT pass is deleted in
    // other.pass_builder(), it will set again, so we just remove the
    // deleted_pass.
    pass_builder_->ClearPasses();
    auto other_passes = other.pass_builder()->AllPasses();
    for (auto pass : other_passes) {
      pass_builder_->AppendPass(pass);
    }
  }
  if (use_dlnne_) {
    auto all_passes = kDlnneSubgraphPasses;
    auto other_passes = other.pass_builder()->AllPasses();
    // We should sort them, because the user may call the SwitchIrDebug
    // interface, which will change the pass.
    std::sort(all_passes.begin(), all_passes.end());
    std::sort(other_passes.begin(), other_passes.end());
    std::vector<std::string> deleted_passes;
    std::set_difference(all_passes.begin(),
                        all_passes.end(),
                        other_passes.begin(),
                        other_passes.end(),
                        std::inserter(deleted_passes, deleted_passes.begin()));
    for (auto ps : deleted_passes) {
      pass_builder_->DeletePass(ps);
    }
  }

  for (auto &delete_pass : other.pass_builder()->GetAllDeletedPasses()) {
    pass_builder_->DeletePass(delete_pass);
  }
}

void AnalysisConfig::EnableCUDNN() {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  use_cudnn_ = use_gpu_;
#else
  LOG(ERROR) << "Please compile with CUDA first to use cuDNN";
  use_cudnn_ = false;
#endif

  Update();
}

void AnalysisConfig::EnableMKLDNN() {
#ifdef PADDLE_WITH_MKLDNN
  use_mkldnn_ = true;
#else
  LOG(ERROR) << "Please compile with MKLDNN first to use MKLDNN";
  use_mkldnn_ = false;
#endif

  Update();
}

void AnalysisConfig::SetMkldnnCacheCapacity(int capacity) {
#ifdef PADDLE_WITH_MKLDNN
  mkldnn_cache_capacity_ = capacity;
#else
  LOG(ERROR) << "Please compile with MKLDNN first to set MKLDNN Thread Id";
  mkldnn_cache_capacity_ = 0;
#endif
}

void AnalysisConfig::EnableMkldnnQuantizer() {
#ifdef PADDLE_WITH_MKLDNN
  if (!mkldnn_quantizer_config_)
    mkldnn_quantizer_config_.reset(new MkldnnQuantizerConfig());
  use_mkldnn_quantizer_ = true;
#else
  LOG(ERROR) << "Please compile with MKLDNN first to use MkldnnQuantizer";
  use_mkldnn_quantizer_ = false;
#endif

  Update();
}

void AnalysisConfig::EnableMkldnnBfloat16() {
#ifdef PADDLE_WITH_MKLDNN
  if (phi::backends::cpu::MayIUse(phi::backends::cpu::cpu_isa_t::avx512_core)) {
    use_mkldnn_bfloat16_ = true;
    LOG(INFO) << "Hardware support for BFLOAT16"
              << (phi::backends::cpu::MayIUse(
                      phi::backends::cpu::cpu_isa_t::avx512_bf16)
                      ? " is enabled"
                      : " is disabled. Simulation will be used");
  } else {
    LOG(INFO) << "CPU does not support BFLOAT16 calculations";
    use_mkldnn_bfloat16_ = false;
  }
#else
  LOG(ERROR) << "Please compile with MKLDNN first to use MkldnnBfloat16";
  use_mkldnn_bfloat16_ = false;
#endif

  Update();
}

void AnalysisConfig::DisableMkldnnFcPasses() {
#ifdef PADDLE_WITH_MKLDNN
  disable_mkldnn_fc_passes_ = true;
#else
  LOG(ERROR) << "Please compile with MKLDNN first to use DisableMkldnnFcPasses";
  disable_mkldnn_fc_passes_ = false;
#endif
  Update();
}

void AnalysisConfig::EnableMkldnnInt8(
    const std::unordered_set<std::string> &op_list) {
#ifdef PADDLE_WITH_MKLDNN
  use_mkldnn_int8_ = true;
  use_fc_padding_ = false;
  if (!op_list.empty())
    quantize_enabled_op_types_.insert(op_list.begin(), op_list.end());
#else
  LOG(ERROR) << "Please compile with MKLDNN first to use MkldnnInt8";
  use_mkldnn_int8_ = false;
#endif

  Update();
}

MkldnnQuantizerConfig *AnalysisConfig::mkldnn_quantizer_config() const {
  PADDLE_ENFORCE_NOT_NULL(mkldnn_quantizer_config_,
                          platform::errors::PreconditionNotMet(
                              "MkldnnQuantizer was not enabled yet."));
  return mkldnn_quantizer_config_.get();
}

void AnalysisConfig::EnableTensorRtEngine(
    int64_t workspace_size,
    int max_batch_size,
    int min_subgraph_size,
    AnalysisConfig::Precision precision_mode,
    bool use_static,
    bool use_calib_mode) {
#ifdef PADDLE_WITH_TENSORRT
  if (!use_gpu()) {
    LOG(ERROR) << "To use TensorRT engine, please call EnableUseGpu() first";
    return;
  }

  use_tensorrt_ = true;
  tensorrt_workspace_size_ = workspace_size;
  tensorrt_max_batchsize_ = max_batch_size;
  tensorrt_min_subgraph_size_ = min_subgraph_size;
  tensorrt_precision_mode_ = precision_mode;
  trt_use_static_engine_ = use_static;
  trt_use_calib_mode_ = use_calib_mode;

  Update();
#else
  PADDLE_THROW(platform::errors::PreconditionNotMet(
      "To use Paddle-TensorRT, please compile with TENSORRT first."));
#endif
}

void AnalysisConfig::EnableTensorRTMemoryOptim(bool engine_memory_sharing,
                                               int sharing_identifier) {
  PADDLE_ENFORCE_EQ(
      use_tensorrt_,
      true,
      platform::errors::InvalidArgument(
          "To enable TensorRT memory optim, please call "
          "EnableTensorRtEngine or enable_tensorrt_engine first."));
  PADDLE_ENFORCE_GE(sharing_identifier,
                    0,
                    platform::errors::InvalidArgument(
                        "The value of sharing_identifier must be greater "
                        "than or equal to 0."));
  if (!engine_memory_sharing) {
    PADDLE_ENFORCE_EQ(sharing_identifier,
                      0,
                      platform::errors::InvalidArgument(
                          "The value of sharing_identifier must be equal to 0 "
                          "when engine_memory_sharing is false."));
  }
  trt_engine_memory_sharing_ = engine_memory_sharing;
  trt_engine_memory_sharing_identifier_ = sharing_identifier;
}

void AnalysisConfig::EnableDlnne(
    int min_subgraph_size,
    int max_batch_size,
    bool use_static_batch,
    std::string weight_share_mode,
    std::unordered_set<std::string> disable_nodes_by_ouputs,
    std::map<std::string, std::vector<int64_t>> dlnne_input_shape_dict,
    bool use_calib_mode,
    AnalysisConfig::Precision precision_mode) {
  use_dlnne_ = true;
  dlnne_min_subgraph_size_ = min_subgraph_size;
  dlnne_max_batchsize_ = max_batch_size;
  dlnne_use_static_batch_ = use_static_batch;
  dlnne_weight_share_mode_ = weight_share_mode;
  dlnne_disable_nodes_by_outputs_ = disable_nodes_by_ouputs;
  dlnne_input_shape_dict_ = dlnne_input_shape_dict;
  dlnne_use_calib_mode_ = use_calib_mode;
  dlnne_precision_mode_ = precision_mode;
  Update();
}

void AnalysisConfig::SetTRTDynamicShapeInfo(
    std::map<std::string, std::vector<int>> min_input_shape,
    std::map<std::string, std::vector<int>> max_input_shape,
    std::map<std::string, std::vector<int>> optim_input_shape,
    bool disable_trt_plugin_fp16) {
  min_input_shape_ = min_input_shape;
  max_input_shape_ = max_input_shape;
  optim_input_shape_ = optim_input_shape;
  disable_trt_plugin_fp16_ = disable_trt_plugin_fp16;
}

void AnalysisConfig::EnableTensorRtDLA(int dla_core) {
  trt_use_dla_ = true;
  trt_dla_core_ = dla_core;
}

void AnalysisConfig::EnableTensorRtInspector() { trt_use_inspector_ = true; }

void AnalysisConfig::Exp_DisableTensorRtOPs(
    const std::vector<std::string> &ops) {
  trt_disabled_ops_.insert(trt_disabled_ops_.end(), ops.begin(), ops.end());
}

void AnalysisConfig::EnableVarseqlen() { trt_use_varseqlen_ = true; }

// TODO(Superjomn) refactor this, buggy.
void AnalysisConfig::Update() {
  auto &&info = SerializeInfoCache();
  if (info == serialized_info_cache_) return;

  // Transfer pass_builder and copy the existing compatible passes.
  if (!pass_builder_ || ((use_gpu() ^ pass_builder_->use_gpu())) ||
      ((use_xpu() ^ pass_builder_->use_xpu())) ||
      ((use_npu() ^ pass_builder_->use_npu())) ||
      ((use_ipu() ^ pass_builder_->use_ipu())) ||
      ((use_custom_device() ^ pass_builder_->use_custom_device()))) {
    if (use_gpu()) {
      pass_builder_.reset(new GpuPassStrategy);
    } else if (use_ipu()) {
      pass_builder_.reset(new IpuPassStrategy);
    } else if (use_xpu()) {
      PADDLE_ENFORCE_EQ(
          use_gpu(),
          false,
          platform::errors::InvalidArgument(
              "Only one choice can be made between CPU and XPU."));
      pass_builder_.reset(new XpuPassStrategy);
    } else if (use_npu()) {
      PADDLE_ENFORCE_EQ(
          use_gpu(),
          false,
          platform::errors::InvalidArgument(
              "Only one choice can be made between GPU and NPU."));
      pass_builder_.reset(new NpuPassStrategy);
    } else if (use_custom_device()) {
      PADDLE_ENFORCE_EQ(
          use_gpu(),
          false,
          platform::errors::InvalidArgument(
              "Only one choice can be made between GPU and CustomDevice."));
      pass_builder_.reset(new CustomDevicePassStrategy);
    } else {
      pass_builder_.reset(new CpuPassStrategy);
    }

  } else {
    if (use_gpu()) {
      pass_builder_.reset(new GpuPassStrategy(
          *static_cast<GpuPassStrategy *>(pass_builder_.get())));
    } else if (use_ipu()) {
      VLOG(1) << "IpuPassStrategy has been used.";
      pass_builder_.reset(new IpuPassStrategy(
          *static_cast<IpuPassStrategy *>(pass_builder_.get())));
    } else if (use_xpu()) {
      PADDLE_ENFORCE_EQ(
          use_gpu(),
          false,
          platform::errors::InvalidArgument(
              "Only one choice can be made between CPU and XPU."));
      pass_builder_.reset(new XpuPassStrategy(
          *static_cast<XpuPassStrategy *>(pass_builder_.get())));
    } else if (use_npu()) {
      PADDLE_ENFORCE_EQ(
          use_gpu(),
          false,
          platform::errors::InvalidArgument(
              "Only one choice can be made between GPU and NPU."));
      pass_builder_.reset(new NpuPassStrategy(
          *static_cast<NpuPassStrategy *>(pass_builder_.get())));
    } else if (use_custom_device()) {
      PADDLE_ENFORCE_EQ(
          use_gpu(),
          false,
          platform::errors::InvalidArgument(
              "Only one choice can be made between GPU and CustomDevice."));
      pass_builder_.reset(new CustomDevicePassStrategy(
          *static_cast<CustomDevicePassStrategy *>(pass_builder_.get())));
    } else {
      pass_builder_.reset(new CpuPassStrategy(
          *static_cast<CpuPassStrategy *>(pass_builder_.get())));
    }
  }

  if (use_tensorrt_) {
    pass_builder()->ClearPasses();
    for (const auto &pass : kTRTSubgraphPasses) {
      if (tensorrt_precision_mode_ == AnalysisConfig::Precision::kInt8 &&
          (pass == "conv_bn_fuse_pass")) {
        continue;
      }
      pass_builder()->AppendPass(pass);
    }
  }

  // TODO(wilber): An ugly method to update pass, need to be fixed.
  if (use_cinn_compiler_) {
    pass_builder()->ClearPasses();
    for (const auto &pass : kCINNCompilerPasses) {
      pass_builder()->AppendPass(pass);
    }
  }

  if (use_dlnne_) {
    pass_builder()->ClearPasses();
    for (const auto &pass : kDlnneSubgraphPasses) {
      pass_builder()->AppendPass(pass);
    }
  }

  if (use_gpu() && use_cudnn_) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (!enable_ir_optim_) {
      LOG(ERROR) << "EnableCUDNN() only works when IR optimization is enabled.";
    } else {
      pass_builder()->EnableCUDNN();
    }
#endif
  }

  if (use_mkldnn_) {
#ifdef PADDLE_WITH_MKLDNN
    if (!enable_ir_optim_) {
      LOG(ERROR)
          << "EnableMKLDNN() only works when IR optimization is enabled.";
    } else {
      pass_builder()->EnableMKLDNN();
    }
#endif
  }

  // Quantization passes must come after all other optimization passes
  if (use_mkldnn_quantizer_) {
    if (!enable_ir_optim_) {
      LOG(ERROR) << "EnableMkldnnQuantizer() only works when IR optimization "
                    "is enabled.";
    }
#ifdef PADDLE_WITH_MKLDNN
    pass_builder()->EnableMkldnnQuantizer();
#endif
  }

  if (use_mkldnn_bfloat16_) {
#ifdef PADDLE_WITH_MKLDNN
    pass_builder()->EnableMkldnnBfloat16();
#endif
  }

  if (use_mkldnn_int8_) {
#ifdef PADDLE_WITH_MKLDNN
    if (!enable_ir_optim_) {
      LOG(ERROR) << "EnableMkldnnInt8() only works when IR optimization "
                    "is enabled.";
    } else if (!use_mkldnn_) {
      LOG(ERROR) << "EnableMkldnnInt8() only works when MKLDNN "
                    "is enabled.";
    } else {
      pass_builder()->EnableMkldnnInt8();
    }
#endif
  }

  if (disable_mkldnn_fc_passes_) {
#ifdef PADDLE_WITH_MKLDNN
    pass_builder()->DisableMkldnnFcPasses();
#endif
  }

  // TODO(inference): When we enable memory_optimize and mkldnn, PaddleSeg model
  // fail.
  if (enable_memory_optim_) {
#ifdef PADDLE_WITH_MKLDNN
    if (use_mkldnn_) {
      enable_memory_optim_ = false;
      LOG_FIRST_N(WARNING, 1)
          << "It is detected that mkldnn and memory_optimize_pass are enabled "
             "at the same time, but they are not supported yet. Currently, "
             "memory_optimize_pass is explicitly disabled";
    } else {
      pass_builder()->AppendAnalysisPass("memory_optimize_pass");
    }
#else
    pass_builder()->AppendAnalysisPass("memory_optimize_pass");
#endif
  }

  if (use_lite_) {
#ifndef PADDLE_WITH_LITE
    LOG(WARNING) << "You tried to enable the lite subgraph "
                    "but did not have the option -DWITH_LITE compiled.";
#endif
    pass_builder()->ClearPasses();
    for (const auto &pass : kLiteSubgraphPasses) {
      if (std::find(lite_passes_filter_.begin(),
                    lite_passes_filter_.end(),
                    pass) == lite_passes_filter_.end()) {
        pass_builder()->AppendPass(pass);
      }
    }
  }

  if (use_xpu_) {
#if (defined LITE_SUBGRAPH_WITH_XPU) || (defined PADDLE_WITH_XPU)
    PADDLE_ENFORCE_EQ(use_gpu_,
                      false,
                      platform::errors::Unavailable(
                          "Currently, XPU and GPU cannot be enabled in the "
                          "same analysis configuration."));
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "You tried to use an XPU device, but Paddle was not compiled "
        "with XPU-runtime."));
#endif
  }

  if (use_npu_) {
#if defined(PADDLE_WITH_ASCEND_CL) || defined(LITE_SUBGRAPH_WITH_NPU)
    PADDLE_ENFORCE_EQ(use_gpu_,
                      false,
                      platform::errors::Unavailable(
                          "Currently, NPU and GPU cannot be enabled in the "
                          "same analysis configuration."));
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "You tried to use an NPU device, but Paddle was not compiled "
        "with NPU-runtime."));
#endif
  }
  if (use_ipu_) {
#ifndef PADDLE_WITH_IPU
    PADDLE_THROW(platform::errors::Unavailable(
        "You tried to enable the ipu "
        "but did not have the option -DWITH_IPU compiled."));
#endif
  }
  if (use_custom_device_) {
#ifndef PADDLE_WITH_CUSTOM_DEVICE
    PADDLE_THROW(platform::errors::Unavailable(
        "You tried to enable the custom device "
        "but did not have the option -DWITH_CUSTOM_DEVICE compiled."));
#endif
  }
}

std::string AnalysisConfig::SerializeInfoCache() {
  std::stringstream ss;
  ss << model_dir_;
  ss << prog_file_;
  ss << params_file_;

  ss << use_gpu_;
  ss << enable_gpu_mixed_;
  ss << use_external_stream_;
  ss << exec_stream_;
  ss << use_fc_padding_;
  ss << gpu_device_id_;
  ss << xpu_device_id_;
  ss << memory_pool_init_size_mb_;

  ss << use_tensorrt_;
  ss << tensorrt_workspace_size_;
  ss << tensorrt_max_batchsize_;
  ss << tensorrt_min_subgraph_size_;

  ss << use_dlnne_;
  ss << dlnne_min_subgraph_size_;

  for (auto &op : trt_disabled_ops_) ss << op.c_str();
  ss << ";";

  ss << trt_use_dla_;
  ss << trt_dla_core_;

  ss << enable_memory_optim_;
  ss << trt_engine_memory_sharing_;

  ss << use_mkldnn_;
  ss << mkldnn_cache_capacity_;
  for (auto &item : mkldnn_enabled_op_types_) ss << item;
  ss << ";";

  ss << use_mkldnn_quantizer_;
  ss << use_mkldnn_bfloat16_;
  for (auto &item : bfloat16_enabled_op_types_) ss << item;
  ss << use_mkldnn_int8_;
  for (auto &item : quantize_enabled_op_types_) ss << item;
  for (auto &item : quantize_excluded_op_ids_) ss << item;
  ss << ";";
  ss << model_from_memory_;

  ss << with_profile_;

  ss << with_glog_info_;

  ss << enable_ir_optim_;
  ss << use_feed_fetch_ops_;
  ss << ir_debug_;

  ss << specify_input_name_;
  ss << cpu_math_library_num_threads_;

  ss << use_lite_;
  ss << use_xpu_;
  ss << xpu_l3_workspace_size_;
  ss << xpu_locked_;
  ss << xpu_autotune_;
  ss << xpu_autotune_file_;
  ss << xpu_precision_;
  ss << xpu_adaptive_seqlen_;
  ss << xpu_enable_multi_stream_;

  ss << use_npu_;
  ss << npu_device_id_;

  ss << thread_local_stream_;

  ss << use_ipu_;
  ss << ipu_device_num_;
  ss << ipu_micro_batch_size_;
  ss << ipu_enable_pipelining_;
  ss << ipu_batches_per_step_;
  ss << ipu_enable_fp16_;
  ss << ipu_replica_num_;
  ss << ipu_available_memory_proportion_;
  ss << ipu_enable_half_partial_;
  ss << ipu_enable_model_runtime_executor_;
  for (auto custom_op : ipu_custom_ops_info_)
    for (auto attr : custom_op) ss << attr;
  ss << ";";
  for (auto pattern : ipu_custom_patterns_)
    for (auto attr : pattern) ss << attr;
  ss << ";";
  for (auto &op : mixed_black_list_) ss << op.c_str();
  return ss.str();
}

void AnalysisConfig::SetCpuMathLibraryNumThreads(
    int cpu_math_library_num_threads) {
  cpu_math_library_num_threads_ = cpu_math_library_num_threads;

  Update();
}

float AnalysisConfig::fraction_of_gpu_memory_for_pool() const {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // Get the GPU memory details and calculate the fraction of memory for the
  // GPU memory pool.
  size_t gpu_total, gpu_available;
  platform::SetDeviceId(gpu_device_id_);
  platform::GpuMemoryUsage(&gpu_available, &gpu_total);
  double total_gpu_memory = gpu_total / 1024. / 1024.;
  float fraction_of_gpu_memory =
      static_cast<double>(memory_pool_init_size_mb()) / total_gpu_memory;
  VLOG(3) << "total_gpu_memory is " << total_gpu_memory
          << "M, gpu_available is " << gpu_available / 1024. / 1024.
          << "M, memory_pool_init_size is " << memory_pool_init_size_mb()
          << "M.";
  return fraction_of_gpu_memory;
#else
  return 0.;
#endif
}

void AnalysisConfig::EnableMemoryOptim(bool x) {
  enable_memory_optim_ = x;
  Update();
}

bool AnalysisConfig::enable_memory_optim() const {
  return enable_memory_optim_;
}

bool AnalysisConfig::trt_engine_memory_sharing() const {
  return trt_engine_memory_sharing_;
}

void AnalysisConfig::SetModelBuffer(const char *prog_buffer,
                                    size_t prog_buffer_size,
                                    const char *param_buffer,
                                    size_t param_buffer_size) {
  prog_file_ = std::string(prog_buffer, prog_buffer + prog_buffer_size);
  params_file_ = std::string(param_buffer, param_buffer + param_buffer_size);
  model_from_memory_ = true;
}

NativeConfig AnalysisConfig::ToNativeConfig() const {
  NativeConfig config;
  config.model_dir = model_dir_;
  config.prog_file = prog_file_;
  config.param_file = params_file_;
  config.use_gpu = use_gpu_;
  config.device = gpu_device_id_;
  config.fraction_of_gpu_memory = fraction_of_gpu_memory_for_pool();
  config.specify_input_name = specify_input_name_;
  return config;
}

void AnalysisConfig::SwitchIrDebug(int x) {
  ir_debug_ = x;
  Update();
}

void AnalysisConfig::EnableProfile() {
  with_profile_ = true;
  Update();
}

void AnalysisConfig::DisableGlogInfo() {
  with_glog_info_ = false;
  Update();
}

void AnalysisConfig::EnableLiteEngine(
    AnalysisConfig::Precision precision_mode,
    bool zero_copy,
    const std::vector<std::string> &passes_filter,
    const std::vector<std::string> &ops_filter) {
  use_lite_ = true;
  lite_precision_mode_ = precision_mode;
  lite_passes_filter_ = passes_filter;
  lite_ops_filter_ = ops_filter;
  lite_zero_copy_ = zero_copy;
  Update();
}

void AnalysisConfig::EnableOpenCL() {
  use_opencl_ = true;
  Update();
}

void AnalysisConfig::PartiallyRelease() {
  prog_file_.clear();
  prog_file_.shrink_to_fit();
  params_file_.clear();
  params_file_.shrink_to_fit();
}

void AnalysisConfig::EnableGpuMultiStream() { thread_local_stream_ = true; }

std::string AnalysisConfig::Summary() {
  const std::vector<std::string> header{"Option", "Value"};
  paddle::inference::TablePrinter os(header);

  if (!model_dir_.empty()) {
    os.InsertRow({"model_dir", model_dir_});
  }
  if (!(prog_file_.empty() && params_file_.empty())) {
    os.InsertRow({"model_file", prog_file_});
    os.InsertRow({"params_file", params_file_});
  }

  if (model_from_memory_) {
    os.InsertRow({"model_from_memory", params_file_});
  }
  os.InsetDivider();

  // cpu info
  os.InsertRow(
      {"cpu_math_thread", std::to_string(cpu_math_library_num_threads_)});
  os.InsertRow({"enable_mkldnn", use_mkldnn_ ? "true" : "false"});
  os.InsertRow(
      {"mkldnn_cache_capacity", std::to_string(mkldnn_cache_capacity_)});
  os.InsetDivider();

  // gpu info
  os.InsertRow({"use_gpu", use_gpu_ ? "true" : "false"});
  if (use_gpu_) {
    os.InsertRow({"use_cutlass", use_cutlass_ ? "true" : "false"});
    os.InsertRow({"gpu_device_id", std::to_string(gpu_device_id_)});
    os.InsertRow({"enable_gpu_mixed", std::to_string(enable_gpu_mixed_)});
    os.InsertRow({"memory_pool_init_size",
                  std::to_string(memory_pool_init_size_mb_) + "MB"});
    os.InsertRow(
        {"use_external_stream", use_external_stream_ ? "true" : "false"});
    os.InsertRow(
        {"thread_local_stream", thread_local_stream_ ? "true" : "false"});

    os.InsertRow({"use_tensorrt", use_tensorrt_ ? "true" : "false"});
    if (use_tensorrt_) {
#ifdef PADDLE_WITH_TENSORRT
      auto Precision2String =
          [](paddle::AnalysisConfig::Precision prec) -> std::string {
        if (prec == Precision::kFloat32)
          return "fp32";
        else if (prec == Precision::kHalf)
          return "fp16";
        else if (prec == Precision::kInt8)
          return "int8";
        else
          return "None";
      };
      auto version2string =
          [](const std::tuple<int, int, int> &ver) -> std::string {
        std::ostringstream os;
        int major = std::get<0>(ver);
        int minor = std::get<1>(ver);
        int patch = std::get<2>(ver);
        os << major << "." << minor << "." << patch;
        return os.str();
      };
      os.InsertRow(
          {"trt_compile_version",
           version2string(inference::tensorrt::GetTrtCompileVersion())});
      os.InsertRow(
          {"trt_runtime_version",
           version2string(inference::tensorrt::GetTrtRuntimeVersion())});
      os.InsertRow({"tensorrt_precision_mode",
                    Precision2String(tensorrt_precision_mode_)});
      os.InsertRow({"tensorrt_workspace_size",
                    std::to_string(tensorrt_workspace_size_)});
      os.InsertRow(
          {"tensorrt_max_batch_size", std::to_string(tensorrt_max_batchsize_)});
      os.InsertRow({"tensorrt_min_subgraph_size",
                    std::to_string(tensorrt_min_subgraph_size_)});
      os.InsertRow({"tensorrt_use_static_engine",
                    trt_use_static_engine_ ? "true" : "false"});
      os.InsertRow(
          {"tensorrt_use_calib_mode", trt_use_calib_mode_ ? "true" : "false"});

      // dynamic_shape
      os.InsertRow({"tensorrt_enable_dynamic_shape",
                    min_input_shape_.empty() ? "false" : "true"});
      os.InsertRow(
          {"tensorrt_tuned_dynamic_shape",
           trt_tuned_dynamic_shape_ ? shape_range_info_path_ : "false"});

      os.InsertRow(
          {"tensorrt_use_varseqlen", trt_use_varseqlen_ ? "true" : "false"});
      os.InsertRow({"tensorrt_with_interleaved",
                    trt_with_interleaved_ ? "true" : "false"});
      os.InsertRow({"tensorrt_transformer_posid", tensorrt_transformer_posid_});
      os.InsertRow(
          {"tensorrt_transformer_maskid", tensorrt_transformer_maskid_});
      os.InsertRow({"tensorrt_use_dla", trt_use_dla_ ? "true" : "false"});
      if (trt_use_dla_) {
        os.InsertRow({"tensorrt_dla_core", std::to_string(trt_dla_core_)});
      }
      os.InsertRow({"trt_engine_memory_sharing",
                    trt_engine_memory_sharing_ ? "true" : "false"});
#endif
    }
  }
  os.InsetDivider();

  // xpu info
  os.InsertRow({"use_xpu", use_xpu_ ? "true" : "false"});
  if (use_xpu_) {
    os.InsertRow({"xpu_device_id", std::to_string(xpu_device_id_)});
    os.InsertRow(
        {"xpu_l3_workspace_size", std::to_string(xpu_l3_workspace_size_)});
  }
  os.InsetDivider();

  if (use_lite_) {
    os.InsertRow({"use_lite", use_lite_ ? "true" : "false"});
  }

  // cinn compiler
  os.InsertRow({"use_cinn_compiler", use_cinn_compiler_ ? "true" : "false"});

  // ir info
  os.InsertRow({"ir_optim", enable_ir_optim_ ? "true" : "false"});
  os.InsertRow({"ir_debug", ir_debug_ ? "true" : "false"});
  os.InsertRow({"memory_optim", enable_memory_optim_ ? "true" : "false"});
  os.InsertRow({"enable_profile", with_profile_ ? "true" : "false"});
  os.InsertRow({"enable_log", with_glog_info_ ? "true" : "false"});
  os.InsertRow({"collect_shape_range_info",
                collect_shape_range_info_ ? shape_range_info_path_ : "false"});

  return os.PrintTable();
}

LiteNNAdapterConfig &LiteNNAdapterConfig::SetDeviceNames(
    const std::vector<std::string> &names) {
  nnadapter_device_names = names;
  return *this;
}

LiteNNAdapterConfig &LiteNNAdapterConfig::SetContextProperties(
    const std::string &properties) {
  nnadapter_context_properties = properties;
  return *this;
}

LiteNNAdapterConfig &LiteNNAdapterConfig::SetModelCacheDir(
    const std::string &dir) {
  nnadapter_model_cache_dir = dir;
  return *this;
}

LiteNNAdapterConfig &LiteNNAdapterConfig::SetModelCacheBuffers(
    const std::string &model_cache_token,
    const std::vector<char> &model_cache_buffer) {
  PADDLE_ENFORCE_EQ(model_cache_token.empty(),
                    false,
                    platform::errors::InvalidArgument(
                        "model_cache_token should not be empty."));
  PADDLE_ENFORCE_EQ(model_cache_buffer.empty(),
                    false,
                    platform::errors::InvalidArgument(
                        "model_cache_buffer should not be empty."));
  PADDLE_ENFORCE_EQ(nnadapter_model_cache_buffers.count(model_cache_token),
                    false,
                    platform::errors::InvalidArgument(
                        "model_cache_token has already been set."));

  nnadapter_model_cache_buffers[model_cache_token] = model_cache_buffer;
  return *this;
}

LiteNNAdapterConfig &LiteNNAdapterConfig::SetSubgraphPartitionConfigPath(
    const std::string &path) {
  nnadapter_subgraph_partition_config_path = path;
  return *this;
}

LiteNNAdapterConfig &LiteNNAdapterConfig::SetSubgraphPartitionConfigBuffer(
    const std::string &buffer) {
  nnadapter_subgraph_partition_config_buffer = buffer;
  return *this;
}
LiteNNAdapterConfig &LiteNNAdapterConfig::Enable() {
  use_nnadapter = true;
  return *this;
}
LiteNNAdapterConfig &LiteNNAdapterConfig::Disable() {
  use_nnadapter = false;
  return *this;
}

void AnalysisConfig::CollectShapeRangeInfo(
    const std::string &shape_range_info_path) {
  LOG(INFO) << "In CollectShapeInfo mode, we will disable optimizations and "
               "collect the shape information of "
            << "all intermediate tensors in the compute graph and calculate "
               "the min_shape, max_shape and opt_shape.";
  collect_shape_range_info_ = true;
  PADDLE_ENFORCE_EQ(shape_range_info_path.empty(),
                    false,
                    platform::errors::InvalidArgument(
                        "The shape_range_info_path should not be empty, please "
                        "re-check the argument."));
  shape_range_info_path_ = shape_range_info_path;
}

const std::string &AnalysisConfig::shape_range_info_path() const {
  return shape_range_info_path_;
}

bool AnalysisConfig::shape_range_info_collected() const {
  return collect_shape_range_info_;
}

void AnalysisConfig::EnableTunedTensorRtDynamicShape(
    const std::string &shape_range_info_path, bool allow_build_at_runtime) {
  shape_range_info_path_ = shape_range_info_path;
  trt_allow_build_at_runtime_ = allow_build_at_runtime;
  trt_tuned_dynamic_shape_ = true;
}

bool AnalysisConfig::tuned_tensorrt_dynamic_shape() const {
  return trt_tuned_dynamic_shape_;
}

bool AnalysisConfig::trt_allow_build_at_runtime() const {
  return trt_allow_build_at_runtime_;
}

void AnalysisConfig::Exp_DisableMixedPrecisionOps(
    const std::unordered_set<std::string> &black_list) {
  mixed_black_list_ = black_list;
}

void AnalysisConfig::Exp_EnableCINNCompiler() {
#ifdef PADDLE_WITH_CINN
  use_cinn_compiler_ = true;
  Update();
#else
  PADDLE_THROW(platform::errors::Unavailable(
      "You tried to use CINN compiler, but Paddle was not compiled "
      "with CINN."));
#endif
}

bool AnalysisConfig::cinn_compiler_enabled() const {
  return use_cinn_compiler_;
}

}  // namespace paddle
