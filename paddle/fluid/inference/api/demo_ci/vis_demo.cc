/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/*
 * This file contains demo for mobilenet, se-resnext50 and ocr.
 */

#include <gflags/gflags.h>
#include <glog/logging.h>  // use glog instead of CHECK to avoid importing other paddle header files.
#include "utils.h"  // NOLINT

#ifdef PADDLE_WITH_CUDA
DECLARE_double(fraction_of_gpu_memory_to_use);
#endif
DEFINE_string(modeldir, "", "Directory of the inference model.");
DEFINE_string(refer, "", "path to reference result for comparison.");
DEFINE_string(
    data, "",
    "path of data; each line is a record, format is "
    "'<space splitted floats as data>\t<space splitted ints as shape'");
DEFINE_bool(use_gpu, false, "Whether use gpu.");

namespace paddle {
namespace demo {

using contrib::AnalysisConfig;
/*
 * Use the native and analysis fluid engine to inference the demo.
 */
void Main(bool use_gpu) {
  std::unique_ptr<PaddlePredictor> predictor, analysis_predictor;
  AnalysisConfig config;
  config.param_file = FLAGS_modeldir + "/__params__";
  config.prog_file = FLAGS_modeldir + "/__model__";
  config.use_gpu = use_gpu;
  config.device = 0;
  if (FLAGS_use_gpu) {
    config.fraction_of_gpu_memory = 0.1;  // set by yourself
  }

  VLOG(30) << "init predictor";
  predictor = CreatePaddlePredictor<NativeConfig>(config);
  analysis_predictor = CreatePaddlePredictor<AnalysisConfig>(config);

  VLOG(30) << "begin to process data";
  // Just a single batch of data.
  std::string line;
  std::ifstream file(FLAGS_data);
  std::getline(file, line);
  auto record = ProcessALine(line);
  file.close();

  // Inference.
  PaddleTensor input;
  input.shape = record.shape;
  input.data =
      PaddleBuf(record.data.data(), record.data.size() * sizeof(float));
  input.dtype = PaddleDType::FLOAT32;

  VLOG(30) << "run executor";
  std::vector<PaddleTensor> output, analysis_output;
  predictor->Run({input}, &output, 1);

  VLOG(30) << "output.size " << output.size();
  auto& tensor = output.front();
  VLOG(30) << "output: " << SummaryTensor(tensor);

  // compare with reference result
  CheckOutput(FLAGS_refer, tensor);

  // the analysis_output has some diff with native_output,
  // TODO(luotao): add CheckOutput for analysis_output later.
  analysis_predictor->Run({input}, &analysis_output, 1);
}

}  // namespace demo
}  // namespace paddle

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_use_gpu) {
    paddle::demo::Main(true /*use_gpu*/);
  } else {
    paddle::demo::Main(false /*use_gpu*/);
  }
  return 0;
}
