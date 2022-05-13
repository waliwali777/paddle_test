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

#include "paddle/fluid/operators/collective/send_v2_op.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif
#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/phi/api/include/tensor.h"

namespace paddle {
namespace operators {

template <typename T>
class SendOpV2CUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if (defined(PADDLE_WITH_RCCL) || defined(PADDLE_WITH_NCCL)) && \
    NCCL_VERSION_CODE >= 2703
    int rid = ctx.Attr<int>("ring_id");
    bool dynamic_shape = ctx.Attr<bool>("dynamic_shape");
    PADDLE_ENFORCE_GE(
        rid, 0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for send_v2 op must be non-negative.", rid));

    int peer = ctx.Attr<int>("peer");
    PADDLE_ENFORCE_GE(
        peer, 0,
        platform::errors::InvalidArgument(
            "The peer (%d) for send_v2 op must be non-negative.", peer));
    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(rid)) {
      // Use ProcessGroup
      distributed::ProcessGroup* pg = map->get(rid);
      std::vector<phi::DenseTensor> in_tensor;
      auto x = ctx.Input<framework::LoDTensor>("X");
      in_tensor.push_back(*x);
      auto task = pg->Send(in_tensor, peer);
      return;
    }
    gpuStream_t stream = nullptr;
    auto place = ctx.GetPlace();
    auto comm = platform::NCCLCommContext::Instance().Get(rid, place);
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::CUDADeviceContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }
    PADDLE_ENFORCE_LT(
        peer, comm->nranks(),
        platform::errors::InvalidArgument("The value of peer (%d) you set must "
                                          "be less than comm->nranks (%d).",
                                          peer, comm->nranks()));

    auto* x_var = ctx.InputVar("X");
    if (x_var->IsType<framework::LoDTensorArray>()) {
      PADDLE_ENFORCE_EQ(
          dynamic_shape, false,
          platform::errors::InvalidArgument("Dynamic shape for send/recv not "
                                            "support LoDTensorArray for now."));
      auto& x_array = x_var->Get<framework::LoDTensorArray>();
      for (size_t idx = 0; idx < x_array.size(); idx++) {
        VLOG(3) << "LodTensorArray: idx(" << idx << ")";
        auto& x = x_array.at(idx);
        int numel = x.numel();
        ncclDataType_t dtype =
            platform::ToNCCLDataType(framework::TransToProtoVarType(x.dtype()));
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclSend(
            x.data<T>(), numel, dtype, peer, comm->comm(), stream));
        VLOG(3) << "rank " << comm->rank() << " send " << phi::product(x.dims())
                << " to " << peer;
      }
      return;
    }
    auto x = ctx.Input<framework::LoDTensor>("X");
    int numel = x->numel();

    if (dynamic_shape) {
      VLOG(3) << "send_v2 will use dynamic shape with recv_v2";
      paddle::experimental::DataType shape_dytpe =
          paddle::experimental::DataType::INT64;
      ncclDataType_t nccl_dtype =
          platform::ToNCCLDataType(framework::TransToProtoVarType(shape_dytpe));
      auto dims = x->dims();
      int64_t shape_size = dims.size();

      // step1: send the shape size

      // prepare the shape size tensor on cpu
      framework::Tensor cpu_shape_size_tensor(shape_dytpe);
      cpu_shape_size_tensor.Resize({1});
      cpu_shape_size_tensor.mutable_data(platform::CPUPlace(), shape_dytpe);
      auto* cpu_data = cpu_shape_size_tensor.data<int64_t>();
      cpu_data[0] = shape_size;

      // copy the shape size tensor to gpu and send
      framework::Tensor* gpu_shape_size_tensor =
          new framework::Tensor(shape_dytpe);
      gpu_shape_size_tensor->Resize({1});
      gpu_shape_size_tensor->mutable_data(place, shape_dytpe);
      framework::TensorCopySync(cpu_shape_size_tensor, place,
                                gpu_shape_size_tensor);
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::ncclSend(gpu_shape_size_tensor->data<int64_t>(), 1,
                                      nccl_dtype, peer, comm->comm(), stream));
      VLOG(3) << "send the shape size: " << shape_size << " to peer";

      // step2: send the shape

      // perpare the shape tensor on cpu
      framework::Tensor cpu_shape_tensor(shape_dytpe);
      cpu_shape_tensor.Resize({shape_size});
      cpu_shape_tensor.mutable_data(platform::CPUPlace(), shape_dytpe);
      auto* cpu_shape_data = cpu_shape_tensor.data<int64_t>();
      for (int i = 0; i < shape_size; ++i) {
        cpu_shape_data[i] = dims[i];
      }

      // copy the shape tensor to gpu and send
      framework::Tensor* gpu_shape_tensor = new framework::Tensor(shape_dytpe);
      gpu_shape_tensor->Resize({shape_size});
      gpu_shape_tensor->mutable_data(place, shape_dytpe);
      framework::TensorCopySync(cpu_shape_tensor, place, gpu_shape_tensor);
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclSend(
          gpu_shape_tensor->data<int64_t>(), shape_size, nccl_dtype, peer,
          comm->comm(), stream));
      VLOG(3) << "send the shape: (" << dims << ") to peer";
    }

    ncclDataType_t dtype =
        platform::ToNCCLDataType(framework::TransToProtoVarType(x->dtype()));
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclSend(
        x->data<T>(), numel, dtype, peer, comm->comm(), stream));
    VLOG(3) << "rank " << comm->rank() << " send " << phi::product(x->dims())
            << " to " << peer;
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "PaddlePaddle should be compiled with NCCL "
        "and NCCL version >= 2.7.3 is needed."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(send_v2, ops::SendOpV2CUDAKernel<float>,
                        ops::SendOpV2CUDAKernel<double>,
                        ops::SendOpV2CUDAKernel<int>,
                        ops::SendOpV2CUDAKernel<int64_t>,
                        ops::SendOpV2CUDAKernel<int8_t>,
                        ops::SendOpV2CUDAKernel<plat::float16>);
