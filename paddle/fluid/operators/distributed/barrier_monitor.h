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

#pragma once

#include <gflags/gflags.h>

#include <chrono>  // NOLINT
#include <deque>
#include <functional>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <thread>  // NOLINT

#include <ThreadPool.h>

#include "paddle/fluid/operators/distributed/rpc_server.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace distributed {

enum BarrierType { kSendBarrier, kRecvBarrier };

template <typename T>
class BlockingQueueForBarrier {
 public:
  explicit BlockingQueueForBarrier(size_t capacity) : capacity_(capacity) {
    PADDLE_ENFORCE_GT(capacity_, 0,
                      platform::errors::InvalidArgument(
                          "The capacity must be greater than 0."));
  }

  bool Push(const T &elem) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      worker_cv_.wait(lock, [&] { return queue_.size() < capacity_; });
      queue_.push_back(elem);
    }
    worker_cv_.notify_one();
    return true;
  }

  bool Push(T &&elem) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      worker_cv_.wait(lock, [&] { return queue_.size() < capacity_; });
      queue_.emplace_back(std::move(elem));
    }
    worker_cv_.notify_one();
    return true;
  }

  T Pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    worker_cv_.wait(lock, [=] { return !queue_.empty(); });
    T rc(std::move(queue_.front()));
    queue_.pop_front();
    worker_cv_.notify_one();
    return rc;
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  void ReCapacity(size_t capacity) {
    std::lock_guard<std::mutex> lock(mutex_);
    PADDLE_ENFORCE_GT(capacity_, 0,
                      platform::errors::InvalidArgument(
                          "The capacity must be greater than 0."));
    capacity_ = capacity;
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::deque<T>().swap(queue_);
  }

 private:
  size_t capacity_;
  std::deque<T> queue_;

  mutable std::mutex mutex_;
  std::condition_variable worker_cv_;
};

class BarrierBlock {
 public:
  explicit BarrierBlock(const int id, const BarrierType &type)
      : id_(id), type_(type) {}

  void Wait() {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [this] { return (done); });
  }

  void Done(bool available) {
    std::unique_lock<std::mutex> lck(mutex_);
    available_ = available;
    done_ = true;
    cv_.notify_all();
  }

 private:
  const int id_;
  const BarrierType type_;
  bool done_ = false;
  bool available_ = true;
  std::condition_variable cv_;
  std::mutex mutex_;
};

class BarrierMonitor {
 public:
  explicit BarrierMonitor(int workers)
      : BarrierMonitor(workers, BarrierType::kRecvBarrier, kMaxWaitMS) {}

  explicit BarrierMonitor(int workers, BarrierType type, int64_t max_wait_times)
      : workers_(workers), barrier_type(type), max_wait_ms(max_wait_times) {
    PADDLE_ENFORCE_GT(workers, 0, platform::errors::InvalidArgument(
                                      "trainers must have one or more"));

    send_barrier_queue = std::make_shared<
        BlockingQueueForBarrier<std::shared_ptr<BarrierBlock>>>(workers);
    recv_barrier_queue = std::make_shared<
        BlockingQueueForBarrier<std::shared_ptr<BarrierBlock>>>(workers);

    running_ = true;
    monitor_thread_.reset(
        new std::thread(std::bind(&BarrierMonitor::Monitor, this)));
  }

  static BarrierMonitor *Init(int workers) {
    InitImpl(workers);
    return GetInstance();
  }

  static BarrierMonitor *GetInstance() { return monitor_.get(); }

  bool IncreaseBarrier(const int worker_id, const std::string &barrier);

  void DecreaseWorker();

  int GetWorkerNum() { return workers_; }

  void Monitor();

  void Exchange(bool available);

  void Stop();

  bool IsReady();

  void WaitServerWeakup();

  void ServerWeakup();

  void NotifyWorker(BarrierType type, bool available);

  void Reset(int workers, BarrierType type);

 private:
  // Init is called by GetInstance.
  static void InitImpl(int workers) {
    monitor_.reset(new BarrierMonitor(workers));
  }

  static std::once_flag init_flag_;
  static std::unique_ptr<BarrierMonitor> monitor_;

  std::atomic<int> workers_{0};
  bool running_ = false;

  std::condition_variable server_cv_;
  std::mutex server_mutex_;

  BarrierType barrier_type;
  int64_t max_wait_ms;
  std::unique_ptr<std::thread> monitor_thread_{nullptr};
  std::shared_ptr<BlockingQueueForBarrier<std::shared_ptr<BarrierBlock>>>
      send_barrier_queue;
  std::shared_ptr<BlockingQueueForBarrier<std::shared_ptr<BarrierBlock>>>
      recv_barrier_queue;

  friend class BarrierBlock;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
