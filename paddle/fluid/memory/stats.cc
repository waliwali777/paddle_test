/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/memory/stats.h"

#include "paddle/fluid/memory/allocation/spin_lock.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace memory {

class StatRegistry {
 public:
  static StatRegistry* GetInstance() {
    static StatRegistry instance;
    return &instance;
  }

  StatBase* GetStat(const std::string& stat_type, int dev_id) {
    auto it = stat_map_.find(GetStatKey(stat_type, dev_id));
    if (it == stat_map_.end()) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The STAT type \"%s\" for device %d has not been regeistered.",
          stat_type.c_str(), dev_id));
    }
    return it->second;
  }

  std::string GetStatKey(const std::string& stat_type, int dev_id) {
    return "STAT_Device" + std::to_string(dev_id) + "_" + stat_type;
  }

  int64_t GetCurrentValue(const std::string& stat_type, int dev_id) {
    return GetStat(stat_type, dev_id)->GetCurrentValue();
  }

  int64_t GetPeakValue(const std::string& stat_type, int dev_id) {
    return GetStat(stat_type, dev_id)->GetPeakValue();
  }

  void Register(const std::string& stat_type, int dev_id, StatBase* stat) {
    std::lock_guard<SpinLock> lock_guard(stat_map_lock_);
    stat_map_[GetStatKey(stat_type, dev_id)] = stat;
  }

  void Unregister(const std::string& stat_type, int dev_id) {
    std::lock_guard<SpinLock> lock_guard(stat_map_lock_);
    stat_map_.erase(GetStatKey(stat_type, dev_id));
  }

  void Update(const std::string& stat_type, int dev_id, int64_t increment) {
    stat_map_[GetStatKey(stat_type, dev_id)]->Update(increment);
  }

 private:
  StatRegistry() = default;

  DISABLE_COPY_AND_ASSIGN(StatRegistry);

  std::unordered_map<std::string, StatBase*> stat_map_;
  SpinLock stat_map_lock_;
};

int64_t StatGetCurrentValue(const std::string& stat_type, int dev_id) {
  return StatRegistry::GetInstance()->GetCurrentValue(stat_type, dev_id);
}

int64_t StatGetPeakValue(const std::string& stat_type, int dev_id) {
  return StatRegistry::GetInstance()->GetPeakValue(stat_type, dev_id);
}

void StatUpdate(const std::string& stat_type, int dev_id, int64_t increment) {
  StatRegistry::GetInstance()->Update(stat_type, dev_id, increment);
}

#define MEMORY_STAT_REGISTER_WITH_ID(item, id) \
  StatRegistry::GetInstance()->Register(       \
      #item, id, Stat<ThreadLocalStatDevice##id##item>::GetInstance());

#define MEMORY_STAT_REGISTER(item)        \
  MEMORY_STAT_REGISTER_WITH_ID(item, 0);  \
  MEMORY_STAT_REGISTER_WITH_ID(item, 1);  \
  MEMORY_STAT_REGISTER_WITH_ID(item, 2);  \
  MEMORY_STAT_REGISTER_WITH_ID(item, 3);  \
  MEMORY_STAT_REGISTER_WITH_ID(item, 4);  \
  MEMORY_STAT_REGISTER_WITH_ID(item, 5);  \
  MEMORY_STAT_REGISTER_WITH_ID(item, 6);  \
  MEMORY_STAT_REGISTER_WITH_ID(item, 7);  \
  MEMORY_STAT_REGISTER_WITH_ID(item, 8);  \
  MEMORY_STAT_REGISTER_WITH_ID(item, 9);  \
  MEMORY_STAT_REGISTER_WITH_ID(item, 10); \
  MEMORY_STAT_REGISTER_WITH_ID(item, 11); \
  MEMORY_STAT_REGISTER_WITH_ID(item, 12); \
  MEMORY_STAT_REGISTER_WITH_ID(item, 13); \
  MEMORY_STAT_REGISTER_WITH_ID(item, 14); \
  MEMORY_STAT_REGISTER_WITH_ID(item, 15)

int RegisterAllStats() {
  MEMORY_STAT_REGISTER(Allocated);
  MEMORY_STAT_REGISTER(Reserved);
  return 0;
}

UNUSED static int regiester_all_stats = RegisterAllStats();

class PrintMem {
 public:
  PrintMem() {
    MEMORY_STAT_UPDATE(Reserved, 0, 0);
    MEMORY_STAT_UPDATE(Reserved, 1, 0);
    MEMORY_STAT_UPDATE(Reserved, 2, 0);
    MEMORY_STAT_UPDATE(Reserved, 3, 0);
    MEMORY_STAT_UPDATE(Reserved, 4, 0);
    MEMORY_STAT_UPDATE(Reserved, 5, 0);
    MEMORY_STAT_UPDATE(Reserved, 6, 0);
    MEMORY_STAT_UPDATE(Reserved, 7, 0);

    MEMORY_STAT_UPDATE(Allocated, 0, 0);
    MEMORY_STAT_UPDATE(Allocated, 1, 0);
    MEMORY_STAT_UPDATE(Allocated, 2, 0);
    MEMORY_STAT_UPDATE(Allocated, 3, 0);
    MEMORY_STAT_UPDATE(Allocated, 4, 0);
    MEMORY_STAT_UPDATE(Allocated, 5, 0);
    MEMORY_STAT_UPDATE(Allocated, 6, 0);
    MEMORY_STAT_UPDATE(Allocated, 7, 0);
  }
  ~PrintMem() {
    std::cout << "GPU 0: allocated = "
              << MEMORY_STAT_PEAK_VALUE(Allocated, 0) / 1048576.0
              << "MB, reserved = "
              << MEMORY_STAT_PEAK_VALUE(Reserved, 0) / 1048576.0 << "MB"
              << std::endl;
    std::cout << "GPU 1: allocated = "
              << MEMORY_STAT_PEAK_VALUE(Allocated, 1) / 1048576.0
              << "MB, reserved = "
              << MEMORY_STAT_PEAK_VALUE(Reserved, 1) / 1048576.0 << "MB"
              << std::endl;
    std::cout << "GPU 2: allocated = "
              << MEMORY_STAT_PEAK_VALUE(Allocated, 2) / 1048576.0
              << "MB, reserved = "
              << MEMORY_STAT_PEAK_VALUE(Reserved, 2) / 1048576.0 << "MB"
              << std::endl;
    std::cout << "GPU 3: allocated = "
              << MEMORY_STAT_PEAK_VALUE(Allocated, 3) / 1048576.0
              << "MB, reserved = "
              << MEMORY_STAT_PEAK_VALUE(Reserved, 3) / 1048576.0 << "MB"
              << std::endl;
    std::cout << "GPU 4: allocated = "
              << MEMORY_STAT_PEAK_VALUE(Allocated, 4) / 1048576.0
              << "MB, reserved = "
              << MEMORY_STAT_PEAK_VALUE(Reserved, 4) / 1048576.0 << "MB"
              << std::endl;
    std::cout << "GPU 5: allocated = "
              << MEMORY_STAT_PEAK_VALUE(Allocated, 5) / 1048576.0
              << "MB, reserved = "
              << MEMORY_STAT_PEAK_VALUE(Reserved, 5) / 1048576.0 << "MB"
              << std::endl;
    std::cout << "GPU 6: allocated = "
              << MEMORY_STAT_PEAK_VALUE(Allocated, 6) / 1048576.0
              << "MB, reserved = "
              << MEMORY_STAT_PEAK_VALUE(Reserved, 6) / 1048576.0 << "MB"
              << std::endl;
    std::cout << "GPU 7: allocated = "
              << MEMORY_STAT_PEAK_VALUE(Allocated, 7) / 1048576.0
              << "MB, reserved = "
              << MEMORY_STAT_PEAK_VALUE(Reserved, 7) / 1048576.0 << "MB"
              << std::endl;
  }
};

UNUSED static PrintMem tmp;

}  // namespace memory
}  // namespace paddle
