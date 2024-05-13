/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#include <rccl/rccl.h>

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

#ifdef __cplusplus
extern "C" {
#endif
ncclResult_t ncclCommInitRank2(ncclComm_t* newcomm,
                               int nranks,
                               ncclUniqueId commId,
                               int myrank,
                               int param);
#ifdef __cplusplus
}
#endif

namespace phi {
namespace dynload {

extern std::once_flag rccl_dso_flag;
extern void* rccl_dso_handle;

#define DECLARE_DYNAMIC_LOAD_RCCL_WRAP(__name)                   \
  struct DynLoad__##__name {                                     \
    static auto GetRCCLFunc() {                                  \
      using rccl_func = decltype(&::__name);                     \
      std::call_once(rccl_dso_flag, []() {                       \
        rccl_dso_handle = phi::dynload::GetNCCLDsoHandle();      \
      });                                                        \
      static void* p_##__name = dlsym(rccl_dso_handle, #__name); \
      return reinterpret_cast<rccl_func>(p_##__name);            \
    }                                                            \
                                                                 \
    template <typename... Args>                                  \
    auto operator()(Args... args) -> decltype(__name(args...)) { \
      return GetRCCLFunc()(args...);                             \
    }                                                            \
                                                                 \
    static bool IsValid() { return GetRCCLFunc() != nullptr; }   \
  };                                                             \
  extern DynLoad__##__name __name

#define RCCL_RAND_ROUTINE_EACH(__macro) \
  __macro(ncclCommInitAll);             \
  __macro(ncclGetUniqueId);             \
  __macro(ncclCommInitRank);            \
  __macro(ncclCommInitRank2);           \
  __macro(ncclCommAbort);               \
  __macro(ncclCommDestroy);             \
  __macro(ncclCommCount);               \
  __macro(ncclCommCuDevice);            \
  __macro(ncclCommUserRank);            \
  __macro(ncclAllReduce);               \
  __macro(ncclBcast);                   \
  __macro(ncclAllGather);               \
  __macro(ncclGroupStart);              \
  __macro(ncclGroupEnd);                \
  __macro(ncclReduce);                  \
  __macro(ncclReduceScatter);           \
  __macro(ncclCommGetAsyncError);       \
  __macro(ncclGetErrorString);

RCCL_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_RCCL_WRAP)

#if NCCL_VERSION_CODE >= 2212
#define RCCL_RAND_ROUTINE_EACH_AFTER_2212(__macro) __macro(ncclBroadcast);
RCCL_RAND_ROUTINE_EACH_AFTER_2212(DECLARE_DYNAMIC_LOAD_RCCL_WRAP)
#endif

#if NCCL_VERSION_CODE >= 2304
#define RCCL_RAND_ROUTINE_EACH_AFTER_2304(__macro) __macro(ncclGetVersion);
RCCL_RAND_ROUTINE_EACH_AFTER_2304(DECLARE_DYNAMIC_LOAD_RCCL_WRAP)
#endif

#if NCCL_VERSION_CODE >= 2703
#define RCCL_RAND_ROUTINE_EACH_AFTER_2703(__macro) \
  __macro(ncclSend);                               \
  __macro(ncclRecv);
RCCL_RAND_ROUTINE_EACH_AFTER_2703(DECLARE_DYNAMIC_LOAD_RCCL_WRAP)
#endif

#if NCCL_VERSION_CODE >= 21100
#define RCCL_RAND_ROUTINE_EACH_AFTER_21100(__macro) \
  __macro(ncclRedOpCreatePreMulSum);                \
  __macro(ncclRedOpDestroy);
RCCL_RAND_ROUTINE_EACH_AFTER_21100(DECLARE_DYNAMIC_LOAD_RCCL_WRAP)
#endif
}  // namespace dynload
}  // namespace phi
