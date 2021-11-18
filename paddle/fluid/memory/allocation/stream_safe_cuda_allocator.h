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

#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "paddle/fluid/memory/allocation/allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

class StreamSafeCUDAAllocation : public Allocation {
 public:
  StreamSafeCUDAAllocation(AllocationPtr underlying_allocation,
                           cudaStream_t owning_stream);
  void RecordStream(cudaStream_t stream);
  std::shared_ptr<std::set<cudaStream_t>> GetRecordedStreams();

 private:
  AllocationPtr underlying_allocation_;
  cudaStream_t owning_stream_;
  std::shared_ptr<std::set<cudaStream_t>> recorded_streams_;
  std::mutex mutex_;
};

class StreamSafeCUDAAllocator : public Allocator {
 public:
  StreamSafeCUDAAllocator(
      const std::shared_ptr<Allocator> &underlying_allocator,
      const cudaStream_t default_stream);
  bool IsAllocThreadSafe() const override;
  void ProcessEventsAndFree();

 protected:
  Allocation *AllocateImpl(size_t size) override;
  void FreeImpl(Allocation *allocation) override;
  uint64_t ReleaseImpl(const platform::Place &place) override;

 private:
  struct AllocationInfo {
    std::deque<cudaEvent_t> outstanding_events;
    bool can_be_freed{false};
  };

  void CreateEventForAllRecordedStream(
      std::set<cudaStream_t> *recorded_streams,
      std::deque<cudaEvent_t> *outstanding_events);
  void FreeStreamSafeCUDAAllocation(Allocation *allocation);
  std::shared_ptr<AllocationInfo> GetAllocationInfo(Allocation *);

  std::shared_ptr<Allocator> underlying_allocator_;
  cudaStream_t default_stream_;
  std::unordered_map<Allocation *, std::shared_ptr<AllocationInfo>>
      allocation_info_map_;
  mutable std::recursive_mutex mutex_;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
