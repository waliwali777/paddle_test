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

#include "paddle/fluid/lite/api/cxx_api.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/lite/utils/io.h"

namespace paddle {
namespace lite {

void Predictor::SaveModel(const std::string &dir) {
#ifndef LITE_WITH_ARM
  LOG(INFO) << "Save model to " << dir;
  MkDirRecur(dir);
  program_->PersistModel(dir, program_desc_);
#else
  LOG(INFO) << "Save model to ./";
  program_->PersistModel("./", program_desc_);
#endif
}

lite::Tensor *Predictor::GetInput(size_t offset) {
  auto *_feed_list = program_->exec_scope()->FindVar("feed");
  CHECK(_feed_list) << "no feed variable in exec_scope";
  auto *feed_list = _feed_list->GetMutable<std::vector<lite::Tensor>>();
  if (offset >= feed_list->size()) {
    feed_list->resize(offset + 1);
  }
  return &feed_list->at(offset);
}

const lite::Tensor *Predictor::GetOutput(size_t offset) {
  auto *_fetch_list = program_->exec_scope()->FindVar("fetch");
  CHECK(_fetch_list) << "no fatch variable in exec_scope";
  auto &fetch_list = *_fetch_list->GetMutable<std::vector<lite::Tensor>>();
  CHECK_LT(offset, fetch_list.size()) << "offset " << offset << " overflow";
  return &fetch_list.at(offset);
}

void Predictor::Build(const std::string &model_path, const Place &prefer_place,
                      const std::vector<Place> &valid_places) {
  LoadModel(model_path, scope_.get(), &program_desc_);
  Build(program_desc_, prefer_place, valid_places);
}

const framework::proto::ProgramDesc &Predictor::program_desc() const {
  return program_desc_;
}

}  // namespace lite
}  // namespace paddle
