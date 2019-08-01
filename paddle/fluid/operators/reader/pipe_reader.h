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

#pragma once
#include <memory>
#include <vector>

#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"
#include "paddle/fluid/operators/reader/pipe.h"

namespace paddle {
namespace operators {
namespace reader {

class PipeReader : public framework::FileReader {
 public:
  explicit PipeReader(const int pipe_fd, size_t capacity = 64);
  void ReadNext(std::vector<framework::LoDTensor>* out) override;
  ~PipeReader();
  void Shutdown() override;
  void Start() override;

 private:
  void ThreadFunc();

  std::unique_ptr<ReadPipe> pipe_;
  std::unique_ptr<LoDTensorBlockingQueue> queue_;
  std::unique_ptr<std::thread> thread_;
};
}  // namespace reader
}  // namespace operators
}  // namespace paddle
