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
#include "paddle/fluid/inference/analysis/mkldnn_analyzer.h"

namespace paddle {

using inference::analysis::Argument;
using inference::analysis::MKLDNNAnalyzer;
using framework::proto::ProgramDesc;

/* This predictor is based on the AnalysisPredictor and lets one add and keep
 * order of passes to execute depending on whether MKL-DNN is used or not.
 */
class MKLDNNAnalysisPredictor : public AnalysisPredictor {
 public:
  explicit MKLDNNAnalysisPredictor(const MKLDNNAnalysisConfig& config)
      : AnalysisPredictor(config), config_(config) {}

  bool Init(const std::shared_ptr<framework::Scope>& parent_scope) override;

  bool Run(const std::vector<PaddleTensor>& inputs,
           std::vector<PaddleTensor>* output_data,
           int batch_size = -1) override {
    return NativePaddlePredictor::Run(inputs, output_data, batch_size);
  }

  void OptimizeInferenceProgram() override;

 private:
  MKLDNNAnalysisConfig config_;
};

}  // namespace paddle
