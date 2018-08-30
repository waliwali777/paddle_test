#include <gflags/gflags.h>
#include <glog/logging.h>  // use glog instead of PADDLE_ENFORCE to avoid importing other paddle header files.
#include <gtest/gtest.h>
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"

DEFINE_string(infer_model, "", "Directory of the inference model.");
DEFINE_string(infer_data, "", "Path of the dataset.");
DEFINE_int32(batch_size, 1, "batch size.");
DEFINE_int32(repeat, 1, "How many times to repeat run.");

namespace paddle {

template <typename T>
std::string to_string(const std::vector<T>& vec) {
  std::stringstream ss;
  for (const auto& c : vec) {
    ss << c << " ";
  }
  return ss.str();
}

void Main(int batch_size) {
  // Three sequence inputs.
  std::vector<PaddleTensor> input_slots(4);
  // one batch starts
  // data --
  int64_t data0[] = {0, 1, 2};
  for (auto& input : input_slots) {
    input.data.Resize(sizeof(data0) / sizeof(int64_t));
    memcpy(input.data.data(), data0, sizeof(data0));
    input.shape = std::vector<int>({3, 1});
    // dtype --
    input.dtype = PaddleDType::INT64;
    // LoD --
    input.lod = std::vector<std::vector<size_t>>({{0, 3}});
  }

  // shape --
  // Create Predictor --
  NativeConfig config;
  config.model_dir = FLAGS_infer_model;
  config.use_gpu = false;
  auto predictor =
      CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kAnalysis>(config);

  std::vector<PaddleTensor> output_slots;
  CHECK(predictor->Run(input_slots, &output_slots));
  // Get output
  LOG(INFO) << "get outputs " << output_slots.size();
  for (auto& output : output_slots) {
    LOG(INFO) << "output.shape: " << to_string(output.shape);
    // no lod ?
    CHECK_EQ(output.lod.size(), 0UL);
    LOG(INFO) << "output.dtype: " << output.dtype;
    std::stringstream ss;
    for (int i = 0; i < 5; i++) {
      ss << static_cast<float*>(output.data.data())[i] << " ";
    }
    LOG(INFO) << "output.data summary: " << ss.str();
  }
  // one batch ends
}

TEST(text_classification, basic) { Main(FLAGS_batch_size); }

}  // namespace paddle

USE_PASS(fc_fuse_pass);
USE_PASS(seq_concat_fc_fuse_pass);
USE_PASS(fc_lstm_fuse_pass);
USE_PASS(graph_viz_pass);
USE_PASS(infer_clean_graph_pass);
USE_PASS(attention_lstm_fuse_pass);
