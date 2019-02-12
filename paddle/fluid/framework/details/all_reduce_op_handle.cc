//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>

#include "paddle/fluid/framework/details/all_reduce_op_handle.h"
#include "paddle/fluid/framework/details/container_cast.h"
#include "paddle/fluid/framework/details/reduce_and_gather.h"
#include "paddle/fluid/framework/details/variable_visitor.h"
#include "paddle/fluid/platform/profiler.h"

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
#include "sparse.h"
#endif

// asynchronous nccl allreduce or synchronous issue:
// https://github.com/PaddlePaddle/Paddle/issues/15049
DEFINE_bool(
    sync_nccl_allreduce, false,
    "If set true, will call `cudaStreamSynchronize(nccl_stream)`"
    "after allreduce, this mode can get better performance in some scenarios.");

namespace paddle {
namespace framework {
namespace details {

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
AllReduceOpHandle::AllReduceOpHandle(ir::Node *node,
                                     const std::vector<Scope *> &local_scopes,
                                     const std::vector<platform::Place> &places,
                                     const platform::NCCLContextMap *ctxs,
                                     bool is_encoded, int ranks)
    : OpHandleBase(node),
      local_scopes_(local_scopes),
      places_(places),
      nccl_ctxs_(ctxs),
      is_encoded_(is_encoded),
      ranks_(ranks) {
  if (nccl_ctxs_) {
    for (auto &p : places_) {
      this->SetDeviceContext(p, nccl_ctxs_->DevCtx(p));
    }
  }
}
#else
AllReduceOpHandle::AllReduceOpHandle(ir::Node *node,
                                     const std::vector<Scope *> &local_scopes,
                                     const std::vector<platform::Place> &places)
    : OpHandleBase(node), local_scopes_(local_scopes), places_(places) {}
#endif

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
void AllReduceOpHandle::_RunImplEncoded() {
  platform::RecordEvent record_event(Name(), dev_ctxes_.cbegin()->second);

  WaitInputVarGenerated();

  auto in_var_handles = DynamicCast<VarHandle>(this->Inputs());
  auto out_var_handles = DynamicCast<VarHandle>(this->Outputs());
  PADDLE_ENFORCE_EQ(
      in_var_handles.size(), places_.size(),
      "The NoDummyInputSize should be equal to the number of places.");
  PADDLE_ENFORCE_EQ(
      in_var_handles.size(), out_var_handles.size(),
      "The NoDummyInputSize and NoDummyOutputSize should be equal.");

  std::vector<const LoDTensor *> ins;
  std::vector<const LoDTensor *> outs;
  for (size_t i = 0; i < local_scopes_.size(); ++i) {
    auto &local_scope =
        local_scopes_[i]->FindVar(kLocalExecScopeName)->Get<Scope *>();
    auto &in =
        local_scope->FindVar(in_var_handles[i]->name())->Get<LoDTensor>();
    ins.emplace_back(&in);

    auto &out =
        local_scope->FindVar(out_var_handles[i]->name())->Get<LoDTensor>();
    outs.emplace_back(&out);
  }

  PADDLE_ENFORCE(platform::is_gpu_place(ins[0]->place()));
  PADDLE_ENFORCE(platform::is_gpu_place(outs[0]->place()));
  PADDLE_ENFORCE(nccl_ctxs_, "nccl_ctxs should not be nullptr.");

  int dtype = -1;
  size_t in_numel = 0;
  size_t out_numel = 0;
  PADDLE_ENFORCE(ranks_ > 1);
  std::vector<std::function<void()>> all_reduce_calls;
  std::vector<memory::allocation::AllocationPtr> ptrs;
  for (size_t i = 0; i < local_scopes_.size(); ++i) {
    auto &place = places_[i];
    auto &in = *ins[i];
    void *in_tensor_buf = const_cast<void *>(in.data<void>());

    auto &out = *outs[i];
    void *out_tensor_buf = const_cast<void *>(out.data<void>());

    dtype = (dtype == -1) ? platform::ToNCCLDataType(in.type()) : dtype;
    in_numel = (in_numel == 0) ? static_cast<size_t>(in.numel()) : in_numel;
    PADDLE_ENFORCE(in_numel % 2 == 0);
    size_t k = in_numel / 2;
    out_numel = (out_numel == 0) ? static_cast<size_t>(out.numel()) : out_numel;

    int dev_id = boost::get<platform::CUDAPlace>(place).device;
    auto &nccl_ctx = nccl_ctxs_->at(dev_id);
    auto stream = nccl_ctx.stream();
    auto comm = nccl_ctx.comm_;

    auto &allocator =
        platform::DeviceTemporaryAllocator::Instance().Get(place, stream);
    int buf_size = ranks_ * in_numel * sizeof(float);
    auto tmp_ious_data = allocator.Allocate(buf_size);
    void *gather_buff = reinterpret_cast<void *>(tmp_ious_data->ptr());
    ptrs.emplace_back(std::move(tmp_ious_data));

    VLOG(10) << "in_numel:" << in_numel << ", out_numel:" << out_numel
             << ", ranks:" << ranks_ << ", gather_buf size:" << buf_size
             << ", k:" << k << ", place:" << place << ", stream:" << stream
             << ", dtype:" << dtype << ", out_tensor_buf:" << out_tensor_buf;

    all_reduce_calls.emplace_back([=] {
sparseAllGReduce(in_tensor_buf, gather_buff, k, out_tensor_buf, out_numel,
                 static_cast<ncclDataType_t>(dtype), ncclSum, comm,
                 stream);
            /*
      PADDLE_ENFORCE(platform::dynload::ncclAllGather(
          in_tensor_buf, gather_buff, in_numel * sizeof(float), ncclChar, comm,
          stream));
          */
    });
  }

  this->RunAndRecordEvent([&] {
    if (all_reduce_calls.size() == 1UL) {
      // Do not use NCCLGroup when manage NCCL by per thread per device
      all_reduce_calls[0]();
    } else {
      platform::NCCLGroupGuard guard;
      for (auto &call : all_reduce_calls) {
        call();
      }
    }
  });

  if (FLAGS_sync_nccl_allreduce) {
    for (auto &p : places_) {
      int dev_id = boost::get<platform::CUDAPlace>(p).device;
      auto &nccl_ctx = nccl_ctxs_->at(dev_id);
      auto stream = nccl_ctx.stream();
      // cudaStreamSynchronize(stream);
      cudaError_t e_sync = cudaStreamSynchronize(stream);
      if (e_sync != 0) {
        VLOG(10) << "cudaStreamSynchronize " << cudaGetErrorString(e_sync);
      }

      cudaError_t e_get = cudaGetLastError();
      if (e_get != 0) {
        VLOG(10) << "cudaGetLastError  " << cudaGetErrorString(e_get)
                 << " errno:" << e_get;
        exit(-1);
      }
    }
  }
}
#endif

void AllReduceOpHandle::RunImpl() {
  if (!is_encoded_) {
    _RunImpl();
    return;
  }

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  _RunImplEncoded();
#else
  PADDLE_THROW("Not compiled with CUDA");
#endif
}

void AllReduceOpHandle::_RunImpl() {
  platform::RecordEvent record_event(Name(), dev_ctxes_.cbegin()->second);

  WaitInputVarGenerated();
  auto in_var_handles = DynamicCast<VarHandle>(this->Inputs());
  auto out_var_handles = DynamicCast<VarHandle>(this->Outputs());
  PADDLE_ENFORCE_EQ(
      in_var_handles.size(), places_.size(),
      "The NoDummyInputSize should be equal to the number of places.");
  PADDLE_ENFORCE_EQ(
      in_var_handles.size(), out_var_handles.size(),
      "The NoDummyInputSize and NoDummyOutputSize should be equal.");

  std::vector<const LoDTensor *> lod_tensors;
  for (size_t i = 0; i < local_scopes_.size(); ++i) {
    auto *s = local_scopes_[i];
    auto &local_scope = s->FindVar(kLocalExecScopeName)->Get<Scope *>();
    auto &lod_tensor =
        local_scope->FindVar(in_var_handles[i]->name())->Get<LoDTensor>();
    lod_tensors.emplace_back(&lod_tensor);
    VLOG(10) << "place:" << i << ", input_name:" << in_var_handles[i]->name()
             << ", out_name:" << out_var_handles[i]->name();
    PADDLE_ENFORCE_EQ(in_var_handles[i]->name(), out_var_handles[i]->name(),
                      "The name of input and output should be equal.");
  }

  if (platform::is_gpu_place(lod_tensors[0]->place())) {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    PADDLE_ENFORCE(nccl_ctxs_, "nccl_ctxs should not be nullptr.");
    int dtype = -1;
    size_t numel = 0;
    std::vector<std::function<void()>> all_reduce_calls;
    for (size_t i = 0; i < local_scopes_.size(); ++i) {
      auto &p = places_[i];
      auto &lod_tensor = *lod_tensors[i];
      void *buffer = const_cast<void *>(lod_tensor.data<void>());

      if (dtype == -1) {
        dtype = platform::ToNCCLDataType(lod_tensor.type());
      }

      if (numel == 0) {
        numel = static_cast<size_t>(lod_tensor.numel());
      }

      int dev_id = boost::get<platform::CUDAPlace>(p).device;
      auto &nccl_ctx = nccl_ctxs_->at(dev_id);
      auto stream = nccl_ctx.stream();
      auto comm = nccl_ctx.comm_;

      VLOG(10) << "before all reduce buffer:" << buffer << ", numel:" << numel
               << ", dev_id:" << dev_id << ", dtype:" << dtype
               << ", place:" << p;

      all_reduce_calls.emplace_back([=] {
        PADDLE_ENFORCE(platform::dynload::ncclAllReduce(
            buffer, buffer, numel, static_cast<ncclDataType_t>(dtype), ncclSum,
            comm, stream));
      });
    }
    this->RunAndRecordEvent([&] {
      if (all_reduce_calls.size() == 1UL) {
        // Do not use NCCLGroup when manage NCCL by per thread per device
        all_reduce_calls[0]();
      } else {
        platform::NCCLGroupGuard guard;
        for (auto &call : all_reduce_calls) {
          call();
        }
      }
    });

    if (FLAGS_sync_nccl_allreduce) {
      for (auto &p : places_) {
        int dev_id = boost::get<platform::CUDAPlace>(p).device;
        auto &nccl_ctx = nccl_ctxs_->at(dev_id);
        auto stream = nccl_ctx.stream();
        cudaStreamSynchronize(stream);
        // nccl_ctx.Wait();
      }
    }

#else
    PADDLE_THROW("Not compiled with CUDA");
#endif
  } else {  // Special handle CPU only Operator's gradient. Like CRF
    auto &trg = *this->local_scopes_[0]
                     ->FindVar(kLocalExecScopeName)
                     ->Get<Scope *>()
                     ->FindVar(out_var_handles[0]->name())
                     ->GetMutable<framework::LoDTensor>();

    // Reduce All Tensor to trg in CPU
    ReduceLoDTensor func(lod_tensors, &trg);
    VisitDataType(lod_tensors[0]->type(), func);

    for (size_t i = 1; i < local_scopes_.size(); ++i) {
      auto &scope =
          *local_scopes_[i]->FindVar(kLocalExecScopeName)->Get<Scope *>();
      auto &p = places_[i];
      auto *var = scope.FindVar(out_var_handles[i]->name());
      auto *dev_ctx = dev_ctxes_.at(p);

      RunAndRecordEvent(p, [&trg, var, dev_ctx, p] {
        auto &tensor_gpu = *var->GetMutable<framework::LoDTensor>();
        auto &tensor_cpu = trg;
        TensorCopy(tensor_cpu, p, *dev_ctx, &tensor_gpu);
      });
    }
  }
}

std::string AllReduceOpHandle::Name() const { return "all_reduce"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
