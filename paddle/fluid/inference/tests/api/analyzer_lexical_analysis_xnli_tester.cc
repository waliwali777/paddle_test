/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <fstream>
#include <iostream>
#include "paddle/fluid/inference/api/paddle_analysis_config.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"

// setting iterations to 0 means processing the whole dataset
namespace paddle {
namespace inference {
namespace analysis {

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model);
  cfg->DisableGpu();
  cfg->SwitchIrOptim(true);
  cfg->SwitchSpecifyInputNames(false);
  cfg->SetCpuMathLibraryNumThreads(FLAGS_cpu_num_threads);
  cfg->EnableMKLDNN();
}

std::vector<size_t> ReadSentenceLod(std::ifstream &file, size_t offset,
                                    int64_t total_sentences_num) {
  std::vector<size_t> sentence_lod;
  sentence_lod.resize(total_sentences_num);

  file.clear();
  file.seekg(offset);
  file.read(reinterpret_cast<char *>(sentence_lod.data()),
            total_sentences_num * sizeof(size_t));

  if (file.eof()) LOG(ERROR) << "Reached end of stream";
  if (file.fail()) throw std::runtime_error("Failed reading file.");
  return sentence_lod;
}

template <typename T>
class TensorReader {
 public:
  TensorReader(std::ifstream &file, size_t beginning_offset, std::string name)
      : file_(file), position_(beginning_offset), name_(name) {}

  PaddleTensor NextBatch(std::vector<int> shape, std::vector<size_t> lod) {
    int numel =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    PaddleTensor tensor;
    tensor.name = name_;
    tensor.shape = shape;
    tensor.dtype = GetPaddleDType<T>();
    tensor.data.Resize(numel * sizeof(T));
    if (lod.empty() == false) {
      tensor.lod.clear();
      tensor.lod.push_back(lod);
    }
    file_.seekg(position_);
    file_.read(reinterpret_cast<char *>(tensor.data.data()), numel * sizeof(T));
    position_ = file_.tellg();
    if (file_.eof()) LOG(ERROR) << name_ << ": reached end of stream";
    if (file_.fail())
      throw std::runtime_error(name_ + ": failed reading file.");
    return tensor;
  }

 protected:
  std::ifstream &file_;
  size_t position_;
  std::string name_;
};

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs,
              int32_t batch_size = FLAGS_batch_size) {
  std::ifstream file(FLAGS_infer_data, std::ios::binary);
  if (!file) {
    FAIL() << "Couldn't open file: " << FLAGS_infer_data;
  }

  int64_t total_sentences_num = 0L;
  int64_t total_words_num = 0L;
  file.seekg(0);
  file.read(reinterpret_cast<char *>(&total_sentences_num), sizeof(int64_t));
  LOG(INFO) << "Total sentences in file: " << total_sentences_num;
  file.read(reinterpret_cast<char *>(&total_words_num), sizeof(int64_t));
  LOG(INFO) << "Total words in file: " << total_words_num;
  size_t lods_beginning_offset = static_cast<size_t>(file.tellg());
  auto words_begining_offset =
      lods_beginning_offset + sizeof(size_t) * total_sentences_num;
  auto targets_beginning_offset =
      words_begining_offset + sizeof(int64_t) * total_words_num;

  std::vector<size_t> lod_full =
      ReadSentenceLod(file, lods_beginning_offset, total_sentences_num);

  size_t lods_sum = std::accumulate(lod_full.begin(), lod_full.end(), 0UL);
  EXPECT_EQ(lods_sum, static_cast<size_t>(total_words_num));

  TensorReader<int64_t> words_reader(file, words_begining_offset, "words");
  TensorReader<int64_t> targets_reader(file, targets_beginning_offset,
                                       "targets");

  auto iterations_max = total_words_num / batch_size;
  auto iterations = iterations_max;
  if (FLAGS_iterations > 0 && FLAGS_iterations < iterations_max) {
    iterations = FLAGS_iterations;
  }

  for (auto i = 0; i < iterations; i++) {
    // Calculate the words num.  Shape=[words_num, 1]
    std::vector<size_t> batch_lod = {0};
    size_t num_words = 0L;
    std::transform(lod_full.begin() + i * FLAGS_batch_size,
                   lod_full.begin() + (i + 1) * FLAGS_batch_size,
                   std::back_inserter(batch_lod),
                   [&num_words](const size_t lodtemp) -> size_t {
                     num_words += lodtemp;
                     return num_words;
                   });
    auto words_tensor = words_reader.NextBatch(
        {static_cast<int>(batch_lod[FLAGS_batch_size]), 1}, batch_lod);
    if (FLAGS_with_accuracy_layer) {
      auto targets_tensor = targets_reader.NextBatch(
          {static_cast<int>(batch_lod[FLAGS_batch_size]), 1}, batch_lod);
      inputs->emplace_back(std::vector<PaddleTensor>{
          std::move(words_tensor), std::move(targets_tensor)});
    } else {
      inputs->emplace_back(std::vector<PaddleTensor>{std::move(words_tensor)});
    }
  }
}

TEST(Analyzer_lexical_analysis_xnli, quantization) {
  AnalysisConfig config;
  SetConfig(&config);

  std::vector<std::vector<PaddleTensor>> outputs;
  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&config),
                 input_slots_all, &outputs, FLAGS_num_threads,
                 FLAGS_use_analysis);
  // LOG(INFO) << "--- The first dimension of output size is " <<
  // outputs.size();
  // LOG(INFO) << "--- The second dimension of output size is "
  //           << outputs[0].size();
  // int64_t *pdata = static_cast<int64_t *>(outputs[0].data.data());
  // for (size_t i = 0; i < batch1_size; ++i) {
  //   EXPECT_EQ(pdata[i], lac_ref_data[i]);
  // }
  // LOG(INFO) << "--- The value is "<< outputs[0].data.data();
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
