/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class CBroadcastOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto place = ctx.GetPlace();
    PADDLE_ENFORCE(is_gpu_place(place),
                   "CBroadcastOp can run on gpu place only for now.");
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    auto x = ctx.Input<framework::LoDTensor>("X");
    auto out = ctx.Output<framework::LoDTensor>("Out");
    int numel = x->numel();
    ncclDataType_t dtype = platform::ToNCCLDataType(x->type());

    int rid = ctx.Attr<int>("ring_id");
    auto comm = platform::NCCLCommContext::Instance().Get(rid);

    int root = ctx.Attr<int>("root");
    if (root == comm->rank()) {
      PADDLE_ENFORCE(platform::dynload::ncclBcast(
          reinterpret_cast<void*>(const_cast<T*>(x->data<T>())), numel, dtype,
          root, comm->comm(), comm->stream()));
      VLOG(3) << "rank " << comm->rank() << " invoke Bcast. sent "
              << x->numel();

      if (out != x) {
        // TODO(liuyi05): check inplace
        framework::TensorCopy(
            *static_cast<const framework::Tensor*>(x), place,
            *platform::DeviceContextPool::Instance().Get(place),
            static_cast<framework::Tensor*>(out));
      }
    } else {
      PADDLE_ENFORCE(platform::dynload::ncclBcast(
          out->mutable_data<T>(place), numel, dtype, root, comm->comm(),
          comm->stream()));
      VLOG(3) << "rank " << comm->rank() << " invoke Bcast. recieved "
              << framework::product(out->dims());
    }

    out->Resize(x->dims());
    out->set_lod(x->lod());
#else
    PADDLE_THROW("PaddlePaddle should compile with GPU.");
#endif
  }
};

}  // namespace operators
}  // namespace paddle
