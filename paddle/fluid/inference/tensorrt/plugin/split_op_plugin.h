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

#pragma once

#include <vector>
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class SplitPlugin : public PluginTensorRT {
 public:
  SplitPlugin(int axis, std::vector<int> const &output_lengths)
      : axis_(axis), output_length_(output_lengths) {}

  SplitPlugin(void const *serial_data, size_t serial_length) {
    deserializeBase(serial_data, serial_length);
    DeserializeValue(&serial_data, &serial_length, &axis_);
    DeserializeValue(&serial_data, &serial_length, &output_length_);
  }

  SplitPlugin *clone() const override {
    return new SplitPlugin(axis_, output_length_);
  }

  const char *getPluginType() const override { return "split"; }
  int getNbOutputs() const override { return output_length_.size(); }
  nvinfer1::Dims getOutputDimensions(int index,
                                     const nvinfer1::Dims *input_dims,
                                     int num_inputs) override;

  int initialize() override;
  int enqueue(int batchSize, const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override;

 protected:
  size_t getSerializationSize() override {
    return SerializedSize(axis_) + SerializedSize(output_length_) +
           getBaseSerializationSize();
  }

  void serialize(void *buffer) override {
    serializeBase(buffer);
    SerializeValue(&buffer, axis_);
    SerializeValue(&buffer, output_length_);
  }

  int axis_;
  std::vector<int> output_length_;
  int nx_, ny_, nz_;
  std::vector<int> segment_offsets_;
};

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
