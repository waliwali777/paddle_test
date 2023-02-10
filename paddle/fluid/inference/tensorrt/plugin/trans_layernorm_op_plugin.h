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

#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {
class TransLayerNormPluginDynamic : public DynamicPluginTensorRT {
 public:
  TransLayerNormPluginDynamic(const float* bias,
                              const int bias_num,
                              const float* scale,
                              const int scale_num,
                              int begin_norm_axis,
                              float eps,
                              std::vector<int64_t> mean_shape,
                              std::vector<int64_t> variance_shape,
                              bool with_fp16)
      : begin_norm_axis_(begin_norm_axis),
        eps_(eps),
        mean_shape_(mean_shape),
        variance_shape_(variance_shape) {
    with_fp16_ = with_fp16;
    bias_.resize(bias_num);
    scale_.resize(scale_num);
    std::copy(bias, bias + bias_num, bias_.data());
    std::copy(scale, scale + scale_num, scale_.data());
  }

  TransLayerNormPluginDynamic(void const* serialData, size_t serialLength) {
    DeserializeValue(&serialData, &serialLength, &bias_);
    DeserializeValue(&serialData, &serialLength, &scale_);
    DeserializeValue(&serialData, &serialLength, &begin_norm_axis_);
    DeserializeValue(&serialData, &serialLength, &eps_);
    DeserializeValue(&serialData, &serialLength, &mean_shape_);
    DeserializeValue(&serialData, &serialLength, &variance_shape_);
    DeserializeValue(&serialData, &serialLength, &with_fp16_);
  }
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    auto ptr = new TransLayerNormPluginDynamic(bias_.data(),
                                               bias_.size(),
                                               scale_.data(),
                                               scale_.size(),
                                               begin_norm_axis_,
                                               eps_,
                                               mean_shape_,
                                               variance_shape_,
                                               with_fp16_);
    ptr->bias_gpu_ = bias_gpu_;
    ptr->scale_gpu_ = scale_gpu_;
    ptr->fp16_bias_gpu_ = fp16_bias_gpu_;
    ptr->fp16_scale_gpu_ = fp16_scale_gpu_;
    return ptr;
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "trans_layernorm_plugin_dynamic";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 2; }
  int initialize() TRT_NOEXCEPT override;
  void terminate() TRT_NOEXCEPT override;

  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return SerializedSize(bias_) + SerializedSize(scale_) +
           SerializedSize(begin_norm_axis_) + SerializedSize(eps_) +
           SerializedSize(mean_shape_) + SerializedSize(variance_shape_) +
           SerializedSize(with_fp16_);
  }

  void serialize(void* buffer) const TRT_NOEXCEPT override {
    SerializeValue(&buffer, bias_);
    SerializeValue(&buffer, scale_);
    SerializeValue(&buffer, begin_norm_axis_);
    SerializeValue(&buffer, eps_);
    SerializeValue(&buffer, mean_shape_);
    SerializeValue(&buffer, variance_shape_);
    SerializeValue(&buffer, with_fp16_);
  }

  nvinfer1::DimsExprs getOutputDimensions(
      int output_index,
      const nvinfer1::DimsExprs* inputs,
      int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder)  // NOLINT
      TRT_NOEXCEPT override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs,
                                 int nbOutputs) TRT_NOEXCEPT override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) TRT_NOEXCEPT override;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const TRT_NOEXCEPT override {
    return 0;
  }

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs,
              void* const* outputs,
              void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const
      TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override { delete this; }

 private:
  std::vector<float> bias_;
  std::vector<float> scale_;
  phi::DenseTensor mean_t;
  phi::DenseTensor variance_t;
  int begin_norm_axis_;
  float eps_;
  std::vector<int64_t> mean_shape_;
  std::vector<int64_t> variance_shape_;
  // data on devices
  float* bias_gpu_{nullptr};
  float* scale_gpu_{nullptr};
  half* fp16_bias_gpu_{nullptr};
  half* fp16_scale_gpu_{nullptr};
};

class TransLayerNormPluginDynamicCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "trans_layernorm_plugin_dynamic";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    return new TransLayerNormPluginDynamic(serial_data, serial_length);
  }
};

REGISTER_TRT_PLUGIN_V2(TransLayerNormPluginDynamicCreator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
