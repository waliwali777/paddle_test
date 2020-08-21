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

#if defined(PADDLE_WITH_GLOO)
#include <gloo/barrier.h>
#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/file_store.h>
#include <gloo/rendezvous/http_store.h>
#include <gloo/rendezvous/prefix_store.h>
#include <gloo/rendezvous/store.h>
#include <gloo/scatter.h>
#include <gloo/transport/tcp/device.h>
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class CScatterOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_GLOO)
    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");
    auto root_id = ctx.Attr<int>("root");

    int64_t send_numel = in->numel();
    T* send_buff = const_cast<T*>(in->data<T>());
    T* recv_buff = out->data<T>();
    auto place = ctx.GetPlace();
    auto gloo = paddle::framework::GlooWrapper::GetInstance();
    auto nranks = gloo->Size();
    send_numel /= nranks;
    std::vector<T*> ptrs(nranks);
    for (int i = 0; i < nranks; i++) {
      ptrs[i] = send_buff;
      send_buff += send_numel;
    }
    PADDLE_ENFORCE_EQ(
        gloo->IsInitialized(), true,
        platform::errors::InvalidArgument(
            "You must initialize the gloo environment first to use it."));
    gloo::ScatterOptions opts(gloo->GetContext());
    opts.setInputs(ptrs, send_numel);
    opts.setOutput(recv_buff, send_numel);
    opts.setRoot(root_id);

    gloo::scatter(opts);
#else
    PADDLE_ENFORCE_EQ(
        true, false,
        platform::errors::Unavailable(
            "PaddlePaddle should compile with GLOO by setting WITH_GLOO=ON"));
#endif
  }
};

}  // namespace operators
}  // namespace paddle
