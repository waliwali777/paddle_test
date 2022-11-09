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

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/distributed/collective/ProcessGroupStream.h"
#include "paddle/fluid/distributed/store/store.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/device_event.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/device_context.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/NCCLTools.h"
#endif

#ifdef PADDLE_WITH_RCCL
#include "paddle/fluid/platform/dynload/rccl.h"
#else
#include "paddle/fluid/platform/dynload/nccl.h"
#endif

namespace paddle {
namespace distributed {

using Place = paddle::platform::Place;

class ProcessGroupNCCL final : public ProcessGroupStream {
 public:
  class NCCLTask final : public ProcessGroupStream::TaskStream,
                         public std::enable_shared_from_this<NCCLTask> {
   public:
    NCCLTask(const Place& place,
             int rank,
             CommType comm_type,
             bool sync_op,
             bool use_calc_stream);
    virtual ~NCCLTask();

    bool IsCompleted() override;
    bool Wait(std::chrono::milliseconds timeout = kWaitTimeout) override;
    void Synchronize() override;
    void UpdateWaitChain(const phi::DeviceContext& ctx) override;

    bool IsBlockCPUInWait() const { return block_cpu_in_wait_; }
    void SetBlockCPUInWait() { block_cpu_in_wait_ = true; }

    // TODO(sunyilun): methods below will be removed later
    NCCLTask(const std::vector<Place>& places,
             int rank,
             CommType CommType,
             const std::vector<phi::DenseTensor>& inputs);
    NCCLTask(const std::vector<Place>& places,
             int rank,
             CommType comm_type,
             const std::vector<phi::DenseTensor>& inputs,
             bool sync_op,
             bool use_calc_stream);

   private:
    bool block_cpu_in_wait_{false};
    platform::DeviceEvent comm_event_;  // event on comm stream
    Place task_place_;
  };

 public:
  ProcessGroupNCCL(const std::shared_ptr<Store>& store,
                   int rank,
                   int size,
                   const platform::Place& place,
                   int gid);

  std::string GetBackendName() const override { return "NCCL"; }

  const phi::DeviceContext& GetDeviceContext(const Place& place) const override;

  const phi::DeviceContext& GetDeviceContext(
      const Place& place, bool use_calc_stream) const override;

  std::shared_ptr<ProcessGroup::Task> AllGather(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> AllReduce(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const AllreduceOptions& opts,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Barrier(
      const BarrierOptions& = BarrierOptions()) override;

  std::shared_ptr<ProcessGroup::Task> Broadcast(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      const BroadcastOptions& opts,
      bool sync_op,
      bool use_calc_stream) override;

  static void GroupStart();

  static void GroupEnd();

  ncclComm_t NCCLComm(const Place& place) const;

  // TODO(liyurui): This API will be moved later
  std::shared_ptr<ProcessGroup::Task> AllReduce(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const AllreduceOptions& = AllreduceOptions()) override;

  // TODO(sunyilun): methods below will be removed later
  std::shared_ptr<ProcessGroup::Task> Broadcast(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const BroadcastOptions& = BroadcastOptions()) override;

  std::shared_ptr<ProcessGroup::Task> Send(
      std::vector<phi::DenseTensor>& tensors, int dst_rank) override;

  std::shared_ptr<ProcessGroup::Task> Send(
      std::vector<phi::DenseTensor>& tensors,
      int dst_rank,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Recv(
      std::vector<phi::DenseTensor>& tensors, int src_rank) override;

  std::shared_ptr<ProcessGroup::Task> Recv(
      std::vector<phi::DenseTensor>& tensors,
      int src_rank,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Send_Partial(phi::DenseTensor& tensors,
                                                   int dst_rank,
                                                   int64_t offset,
                                                   int64_t length) override;

  std::shared_ptr<ProcessGroup::Task> Send_Partial(
      phi::DenseTensor& tensors,
      int dst_rank,
      int64_t offset,
      int64_t length,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Recv_Partial(phi::DenseTensor& tensors,
                                                   int src_rank,
                                                   int64_t offset,
                                                   int64_t length) override;

  std::shared_ptr<ProcessGroup::Task> Recv_Partial(
      phi::DenseTensor& tensors,
      int src_rank,
      int64_t offset,
      int64_t length,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> AllGather(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors) override;

  std::shared_ptr<ProcessGroup::Task> AllGather_Partial(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      int64_t offset,
      int64_t length) override;

  std::shared_ptr<ProcessGroup::Task> AllGather_Partial(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      int64_t offset,
      int64_t length,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> AllToAll(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors) override;

  std::shared_ptr<ProcessGroup::Task> AllToAll(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> AllToAll_Single(
      std::vector<phi::DenseTensor>& in,
      std::vector<phi::DenseTensor>& out,
      std::vector<int64_t>& in_sizes,
      std::vector<int64_t>& out_sizes) override;

  std::shared_ptr<ProcessGroup::Task> AllToAllSingle(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      std::vector<int64_t>& in_sizes,
      std::vector<int64_t>& out_sizes,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Reduce(
      std::vector<phi::DenseTensor>& tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const ReduceOptions& opts) override;

  std::shared_ptr<ProcessGroup::Task> Reduce(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const ReduceOptions& opts,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> ReduceScatter(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const ReduceScatterOptions& opts,
      bool sync_op,
      bool use_calc_stream) override;

  std::shared_ptr<ProcessGroup::Task> Scatter(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const ScatterOptions& opts) override;

  std::shared_ptr<ProcessGroup::Task> Scatter(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const ScatterOptions& opts,
      bool sync_op,
      bool use_calc_stream) override;

 private:
  std::shared_ptr<ProcessGroupNCCL::NCCLTask> CreateTask(const Place& place,
                                                         int rank,
                                                         CommType op_type,
                                                         bool sync_op,
                                                         bool use_calc_stream);

  void BroadcastUniqueNCCLID(ncclUniqueId* nccl_id);

  void CreateNCCLEnvCache(const Place& place, const std::string& place_key);

  template <typename Fn>
  std::shared_ptr<ProcessGroupStream::Task> Collective(
      phi::DenseTensor* out_tensor,
      const phi::DenseTensor& in_tensor,
      Fn fn,
      CommType comm_type,
      bool sync_op,
      bool use_calc_stream);

  void SyncCalcStream(const Place& place);

  // TODO(sunyilun): methods below will be removed later
  std::shared_ptr<ProcessGroupNCCL::NCCLTask> CreateTask(
      std::vector<Place> places,
      int rank,
      CommType op_type,
      const std::vector<phi::DenseTensor>& inputs);

  std::shared_ptr<ProcessGroupNCCL::NCCLTask> CreateTask(
      const std::vector<Place>& places,
      int rank,
      CommType op_type,
      const std::vector<phi::DenseTensor>& inputs,
      bool sync_op,
      bool use_calc_stream);

  template <typename Fn>
  std::shared_ptr<ProcessGroup::Task> Collective(
      std::vector<phi::DenseTensor>& inputs,   // NOLINT
      std::vector<phi::DenseTensor>& outputs,  // NOLINT
      Fn fn,
      CommType op_type);

  template <typename Fn>
  std::shared_ptr<ProcessGroupStream::Task> Collective(
      std::vector<phi::DenseTensor>& inputs,   // NOLINT
      std::vector<phi::DenseTensor>& outputs,  // NOLINT
      Fn fn,
      CommType comm_type,
      bool sync_op,
      bool use_calc_stream);

  template <typename Fn>
  void Collective(const phi::DenseTensor*,
                  phi::DenseTensor*,
                  Fn fn,
                  CommType op_type);

  template <typename Fn>
  std::shared_ptr<ProcessGroup::Task> PointToPoint(
      std::vector<phi::DenseTensor>& tensors,  // NOLINT
      Fn fn,
      int dst_rank,
      CommType op_type);

  template <typename Fn>
  std::shared_ptr<ProcessGroup::Task> PointToPoint(
      std::vector<phi::DenseTensor>& tensors,  // NOLINT
      Fn fn,
      int dst_rank,
      CommType op_type,
      bool sync_op,
      bool use_calc_stream);

  void CreateNCCLManagerCache(const std::string& places_key,
                              const std::vector<Place>& places);

  void CheckSplitSizes(std::vector<int64_t>* split_sizes,
                       std::vector<int64_t> tensor_shape);

 private:
  std::shared_ptr<Store> store_;
  std::unordered_map<std::string, platform::DeviceEvent>
      place_to_calc_event_;  // event on calc stream
  std::unordered_map<std::string, phi::GPUContext*> place_to_calc_ctx_;
  std::unordered_map<std::string, std::unique_ptr<phi::GPUContext>>
      place_to_comm_ctx_;

  // TODO(sunyilun): attrs below will be removed later
  std::mutex mutex_;
  std::unordered_map<std::string, std::vector<phi::GPUContext*>> places_to_ctx_;
};

}  //  namespace distributed
}  //  namespace paddle
