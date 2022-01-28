// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <iostream>
#include <string>
#include <vector>

namespace paddle {
namespace distributed {

class Store {
 public:
  static constexpr std::chrono::milliseconds kDefaultTimeout =
      std::chrono::seconds(360);
  static constexpr std::chrono::milliseconds kNoTimeout =
      std::chrono::milliseconds::zero();

  Store() : _timeout(kDefaultTimeout) {}
  explicit Store(const std::chrono::milliseconds& timeout)
      : _timeout(timeout) {}
  virtual ~Store() {}

  virtual void set(const std::string& key,
                   const std::vector<uint8_t>& value) = 0;
  virtual std::vector<uint8_t> get(const std::string& key) = 0;

  virtual bool removeKey(const std::string& key) = 0;
  virtual int64_t add(const std::string& key, int64_t value) = 0;
  virtual void wait(const std::vector<std::string>& keys) = 0;
  virtual void wait(const std::vector<std::string>& keys,
                    const std::chrono::milliseconds& timeout) = 0;

  virtual const std::chrono::milliseconds& getTimeout() const {
    return _timeout;
  }
  virtual void setTimeout(const std::chrono::milliseconds& timeout) {
    _timeout = timeout;
  }

 private:
  std::chrono::milliseconds _timeout;
}

}  // namespace distributed
}  // namespace paddle
