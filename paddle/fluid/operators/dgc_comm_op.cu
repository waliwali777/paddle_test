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

#include "dgc/dgc.h"
#include "paddle/fluid/operators/dgc_comm_op.h"
#include "paddle/phi/core/dense_tensor.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif


namespace paddle {
namespace operators {
template <typename DeviceContext, typename T>
class DGCCommOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x = ctx.Input<phi::DenseTensor>("X");
    auto gather = ctx.Input<phi::DenseTensor>("Gather");
    auto grad = ctx.Input<phi::DenseTensor>("Grad");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    
    auto place = ctx.GetPlace();
    out->ShareDataWith(*grad);    

    const int ring_id = ctx.Attr<int>("ring_id");
    const int nranks = ctx.Attr<int>("nranks");
    const int k = ctx.Attr<int>("k_var");
    const int out_numel = static_cast<int>(grad->numel());

    auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
    //auto dev_ctx = platform::DeviceContextPool::Instance().Get(place); 
    auto& dev_ctx = ctx.template device_context<paddle::platform::CUDADeviceContext>();
    
    gpuStream_t stream = comm->stream();

    ncclDataType_t dtype = platform::ToNCCLDataType(framework::TransToProtoVarType(x->dtype()));    

    framework::Tensor gather_res;
    gather_res = ctx.AllocateTmpTensor<T, DeviceContext>(gather->dims(), dev_ctx);

    const T* x_buff = x->data<T>();
    
    T* gather_buff = gather_res.data<T>();
    T* out_data = out->data<T>();

    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
          x_buff, gather_buff, static_cast<int64_t>(2 * k), static_cast<ncclDataType_t>(dtype), comm->comm(), stream));
    
#if defined(PADDLE_WITH_DGC)

    PADDLE_ENFORCE_EQ(paddle::communication::dgc::sparseReduce(
                            (void *)gather_buff, k, out_data, out_numel, nranks, stream),
                        true, platform::errors::Unavailable(
                                  "Calling sparseReduce() failed."));
    //platform::GpuStreamSync(stream);
    //PADDLE_ENFORCE_GPU_SUCCESS(platform::GpuGetLastError());
    
#endif

  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(dgc_comm, ops::DGCCommOpCUDAKernel<paddle::platform::CUDADeviceContext, float>);
