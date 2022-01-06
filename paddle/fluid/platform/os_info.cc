/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/os_info.h"
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>
#if defined(__linux__)
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#elif defined(_MSC_VER)
#include <processthreadsapi.h>
#endif
#include "paddle/fluid/platform/macros.h"  // import DISABLE_COPY_AND_ASSIGN

namespace paddle {
namespace platform {
namespace internal {

template <typename T>
class ThreadDataRegistry {
  class ThreadDataHolder;

 public:
  // Singleton
  static ThreadDataRegistry& GetInstance() {
    static ThreadDataRegistry instance;
    return instance;
  }

  static T& GetCurrentThreadData() { return thread_data_.GetData(); }

  // Returns current snapshot of all threads. Make sure there is no thread
  // create/destory when using it.
  template <typename = std::enable_if_t<std::is_copy_constructible<T>::value>>
  std::unordered_map<uint64_t, T> GetAllThreadDataByValue() {
    std::unordered_map<uint64_t, T> data_copy;
    std::lock_guard<std::mutex> lock(lock_);
    data_copy.reserve(tid_map_.size());
    for (auto& kv : tid_map_) {
      data_copy.emplace(kv.first, kv.second->GetData());
    }
    return std::move(data_copy);
  }

  // Returns current snapshot of all threads. Make sure there is no thread
  // create/destory when using it.
  // std::vector<std::reference_wrapper<T>> GetAllThreadDataByRef() {

  //}

  void RegisterData(uint64_t tid, ThreadDataHolder* tls_obj) {
    std::lock_guard<std::mutex> lock(lock_);
    tid_map_[tid] = tls_obj;
  }

  void UnregisterData(uint64_t tid) {
    std::lock_guard<std::mutex> lock(lock_);
    tid_map_.erase(tid);
  }

 private:
  class ThreadDataHolder {
   public:
    ThreadDataHolder() {
      ThreadDataRegistry::GetInstance().RegisterData(GetCurrentThreadMainId(),
                                                     this);
    }

    ~ThreadDataHolder() {
      ThreadDataRegistry::GetInstance().UnregisterData(
          GetCurrentThreadMainId());
    }

    T& GetData() { return data_; }

   private:
    T data_;
  };

  ThreadDataRegistry() = default;

  DISABLE_COPY_AND_ASSIGN(ThreadDataRegistry);

  static thread_local ThreadDataHolder thread_data_;
  std::mutex lock_;
  std::unordered_map<uint64_t, ThreadDataHolder*> tid_map_;  // not owned
};

template <typename T>
thread_local typename ThreadDataRegistry<T>::ThreadDataHolder
    ThreadDataRegistry<T>::thread_data_;

class ThreadIdHolder {
 public:
  ThreadIdHolder();

  ~ThreadIdHolder();

  const ThreadId& GetTid() const { return id_; }

  uint64_t GetMainTid() const { return id_.MainTid(); }

 private:
  ThreadId id_;
};

class ThreadIdRegistry {
 public:
  // Singleton
  static ThreadIdRegistry& GetInstance() {
    static ThreadIdRegistry instance;
    return instance;
  }

  void RegisterThread(const ThreadIdHolder& id) {
    std::lock_guard<std::mutex> lock(lock_);
    id_map_[id.GetMainTid()] = &id;
  }

  void UnregisterThread(const ThreadIdHolder& id) {
    std::lock_guard<std::mutex> lock(lock_);
    id_map_.erase(id.GetMainTid());
  }

  // Returns current snapshot of all threads.
  std::vector<ThreadId> AllThreadIds();

 private:
  ThreadIdRegistry() = default;

  DISABLE_COPY_AND_ASSIGN(ThreadIdRegistry);

  std::mutex lock_;
  std::unordered_map<uint64_t, const ThreadIdHolder*> id_map_;
};

ThreadIdHolder::ThreadIdHolder() {
  // C++ std tid
  id_.std_tid = std::hash<std::thread::id>()(std::this_thread::get_id());
// system tid
#if defined(__linux__)
  id_.sys_tid = syscall(SYS_gettid);
#elif defined(_MSC_VER)
  id_.sys_tid = GetCurrentThreadId();
#else  // unsupported platforms, use std_tid
  id_.sys_tid = id_.std_tid;
#endif
  // cupti tid
  std::stringstream ss;
  ss << std::this_thread::get_id();
  id_.cupti_tid = static_cast<uint32_t>(std::stoull(ss.str()));
  ThreadIdRegistry::GetInstance().RegisterThread(*this);
}

ThreadIdHolder::~ThreadIdHolder() {
  ThreadIdRegistry::GetInstance().UnregisterThread(*this);
}

std::vector<ThreadId> ThreadIdRegistry::AllThreadIds() {
  std::vector<ThreadId> ids;
  std::lock_guard<std::mutex> lock(lock_);
  ids.reserve(id_map_.size());
  for (const auto& kv : id_map_) {
    ids.push_back(kv.second->GetTid());
  }
  return std::move(ids);
}

static thread_local ThreadIdHolder thread_id;

}  // namespace internal

uint64_t GetCurrentThreadMainId() { return internal::thread_id.GetMainTid(); }

ThreadId GetCurrentThreadId() { return internal::thread_id.GetTid(); }

std::unordered_map<uint64_t, ThreadId> GetAllThreadIds() {
  auto tids = internal::ThreadIdRegistry::GetInstance().AllThreadIds();
  std::unordered_map<uint64_t, ThreadId> res;
  for (const auto& tid : tids) {
    res[tid.MainTid()] = tid;
  }
  return res;
}

static constexpr const char* kDefaultThreadName = "unset";

std::string GetCurrentThreadName() {
  auto& thread_name =
      internal::ThreadDataRegistry<std::string>::GetCurrentThreadData();
  return thread_name.empty() ? kDefaultThreadName : thread_name;
}

std::unordered_map<uint64_t, std::string> GetAllThreadNames() {
  return internal::ThreadDataRegistry<std::string>::GetInstance()
      .GetAllThreadDataByValue();
}

bool SetCurrentThreadName(const std::string& name) {
  auto& thread_name =
      internal::ThreadDataRegistry<std::string>::GetCurrentThreadData();
  if (!thread_name.empty() || name == kDefaultThreadName) {
    return false;
  }
  thread_name = name;
  return true;
}

uint32_t GetProcessId() {
#if defined(_MSC_VER)
  return static_cast<uint32_t>(GetCurrentProcessId());
#else
  return static_cast<uint32_t>(getpid());
#endif
}

}  // namespace platform
}  // namespace paddle
