/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "threadpool.h"
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <map>
#include <thread>

namespace framework = paddle::framework;

void do_sum(framework::ThreadPool* pool, std::atomic<int>& sum, int cnt) {
  for (int i = 0; i < cnt; ++i) {
    pool->Run([&sum]() { sum.fetch_add(1); });
  }
}

TEST(ThreadPool, ConcurrentInit) {
  framework::ThreadPool* pool;
  int concurrent_cnt = 50;
  std::vector<std::thread> threads;
  for (int i = 0; i < concurrent_cnt; ++i) {
    std::thread t([&pool]() { pool = framework::ThreadPool::GetInstance(); });
    threads.push_back(std::move(t));
  }
  for (int i = 0; i < concurrent_cnt; ++i) {
    threads[i].join();
  }
}

TEST(ThreadPool, ConcurrentStart) {
  framework::ThreadPool* pool = framework::ThreadPool::GetInstance();
  std::atomic<int> sum(0);
  int cnt1 = 10, cnt2 = 20;
  std::thread t1(do_sum, pool, std::ref(sum), cnt1);
  std::thread t2(do_sum, pool, std::ref(sum), cnt2);
  t1.join();
  t2.join();
  pool->Wait();
  EXPECT_EQ(sum, cnt1 + cnt2);
}
