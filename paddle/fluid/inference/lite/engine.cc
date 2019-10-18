// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#define LITE_WITH_CUDA 1

#include "paddle/fluid/inference/lite/engine.h"

namespace paddle {
namespace inference {
namespace lite {

bool EngineManager::Empty() const { return engines_.size() == 0; }

bool EngineManager::Has(const std::string& name) const {
  if (engines_.count(name) == 0) {
    return false;
  }
  return engines_.at(name).get() != nullptr;
}

paddle::lite::Predictor* EngineManager::Get(const std::string& name) const {
  return engines_.at(name).get();
}

paddle::lite::Predictor* EngineManager::Create(const std::string& name,
                                               const EngineConfig& cfg) {
  auto* p = new paddle::lite::Predictor();
  p->Build("", cfg.model, cfg.param, cfg.prefer_place, cfg.valid_places,
           cfg.neglected_passes, cfg.model_type, cfg.model_from_memory);
  engines_[name].reset(p);
  return p;
}

void EngineManager::DeleteAll() {
  for (auto& item : engines_) {
    item.second.reset(nullptr);
  }
}

}  // namespace lite
}  // namespace inference
}  // namespace paddle
