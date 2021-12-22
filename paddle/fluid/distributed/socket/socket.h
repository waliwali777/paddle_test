// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <netdb.h>
#include <iostream>
#include <string>

namespace paddle {
namespace distributed {

class Socket {
 public:
  explicit Socket(int sock_fd) : sock_fd(sock_fd) {}
  void listen(const std::string& host, std::uint16_t port);
  void connect(const std::string& host, std::uint16_t port);
  template <typename T>
  void sendValue(const T value);
  template <typename T>
  T recvValue();

  Socket accept() const;

  std::uint16_t getPort() const;

 private:
  int sock_fd;

  bool tryListen(std::uint16_t port);
  bool tryListen(const ::addrinfo& addr);
  bool tryConnect(const std::string& host, std::uint16_t port);
  bool tryConnect(const ::addrinfo& addr);
  template <typename T>
  void sendBytes(const T* buffer, size_t len);
  template <typename T>
  void recvBytes(T* buffer, size_t len);
}

}  // namespace distributed
}  // namespace paddle
