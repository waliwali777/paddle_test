/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "ModelConfig.pb.h"
#include "LayerGradUtil.h"
#include "MkldnnTestFunc.h"

using namespace paddle;  // NOLINT

DECLARE_bool(thread_local_rand_use_global_seed);

struct testFCDesc {
  int bs;
  int ic;
  int oc;
  int ih, iw;  // oh == ow == 1
};

void testFcLayer(const testFCDesc& pm) {
  const std::string compareTypes[] = {"mkldnn_fc", "fc"};
  TestConfig cfg;
  cfg.layerConfig.set_type(compareTypes[0]);
  cfg.layerConfig.set_size(pm.oc);
  cfg.inputDefs.push_back({INPUT_DATA, "layer_0",
    /* size of input layer= */ size_t(pm.ic * pm.ih * pm.iw),
    /* size of weight= */      size_t(pm.oc * pm.ic * pm.ih * pm.iw)});
  LayerInputConfig* input = cfg.layerConfig.add_inputs();
  FCConfig* fc = input->mutable_fc_conf();
  fc->set_dim_in(pm.ic * pm.ih *pm.iw);
  fc->set_dim_out(pm.oc);

  // TODO(TJ): test true and false
  cfg.layerConfig.set_score_with_paddle_wgt(false);

  for (auto biasSize : {pm.oc, 0}) {
    cfg.biasSize = biasSize;
    // test functionality with paddle cpu fc
    TestConfig ref = cfg;
    ref.layerConfig.set_type(compareTypes[1]);
    std::vector<TestConfig> configs = {cfg, ref};
    for (auto bs : {pm.bs, 1}) {
      testLayerFunc(configs, bs, pm.ih, pm.iw);
    }
  }
}

TEST(MkldnnLayer, fcLayer) {
  testFcLayer({2, 2, 3, 1, 1});
  testFcLayer({32, 64, 128, 1, 1});
  testFcLayer({8, 32, 64, 13, 13});
  testFcLayer({8, 32, 32, 13, 11});
  testFcLayer({2, 64, 32, 16, 16});
  testFcLayer({15, 3, 6, 16, 16});
}

// TODO(TJ): add branch test

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}
