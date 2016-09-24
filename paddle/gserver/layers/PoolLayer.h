/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#pragma once

#include "Layer.h"
#include "paddle/math/Matrix.h"
#include <vector>

namespace paddle {

/**
 * @brief Basic parent layer of pooling
 * Pools the input within regions
 */
class PoolLayer : public Layer {
protected:
  size_t channels_, sizeX_, stride_, outputX_, imgSize_;
  int start_, confPadding_;

  size_t sizeY_;
  size_t imgSizeY_;
  size_t strideY_;
  size_t outputY_;
  int confPaddingY_;

  std::string poolType_;

public:
  explicit PoolLayer(const LayerConfig& config) : Layer(config) {}

  /**
   * @brief create pooling layer by pool_type
   */
  static Layer* create(const LayerConfig& config);

  virtual bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  /**
   * Calculate output size according window size and padding size.
   */
  int outputSize(int imageSize, int windowSize, int padding, int stride) {
    int outputSize;
    outputSize =
        (imageSize - windowSize + 2 * padding + stride - 1) / stride + 1;
    return outputSize;
  }
};

}  // namespace paddle
