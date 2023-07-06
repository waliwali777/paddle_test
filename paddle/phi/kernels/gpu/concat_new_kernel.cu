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

#include "paddle/phi/kernels/concat_new_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void ConcatNewKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     int rank,
                     int nranks,
                     int rid,
                     //    bool use_calc_stream,
                     //  bool use_model_parallel,
                     DenseTensor* out) {
  // auto x = ctx.Input<phi::DenseTensor>("X");
  // auto out = ctx.Output<phi::DenseTensor>("Out");
  // ncclDataType_t dtype =
  //     platform::ToNCCLDataType(framework::TransToProtoVarType(x->dtype()));

  // int nranks = ctx.Attr<int>("nranks");
  // int rank = ctx.Attr<int>("rank");
  // int rid = ctx.Attr<int>("ring_id");
  // auto place = ctx.GetPlace();

  // PADDLE_ENFORCE_GE(rank,
  //                   0,
  //                   platform::errors::PreconditionNotMet(
  //                       "The value of rank (%d) for c_concat must be "
  //                       "greater than or equal to 0.",
  //                       rank));
  // PADDLE_ENFORCE_GE(nranks,
  //                   2,
  //                   platform::errors::PreconditionNotMet(
  //                       "The value of nranks (%d) for c_concat must be "
  //                       "greater than or equal to 2.",
  //                       nranks));
  // PADDLE_ENFORCE_LT(rank,
  //                   nranks,
  //                   platform::errors::PreconditionNotMet(
  //                       "The value of rank (%d) for c_concat must be "
  //                       "less than that of nranks (%d).",
  //                       rank,
  //                       nranks));
  // PADDLE_THROW(
  //     errors::PreconditionNotMet("****** in ConcatNewKernel ****** "));
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  auto out_dims = x.dims();
  out_dims[0] *= nranks;
  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);
  auto comm_ctx =
      static_cast<distributed::NCCLCommContext*>(dev_ctx.GetCommContext());
  gpuStream_t stream = dev_ctx.stream();
  comm_ctx->AllGather(out, x, stream);

  // phi::DenseTensor temp_out;
  // framework::DDim temp_out_dims = x->dims();
  // temp_out_dims[0] *= nranks;
  // temp_out.mutable_data<T>(temp_out_dims, place);

  // auto map = distributed::ProcessGroupMapFromGid::getInstance();
  // if (map->has(rid)) {
  //   // Use ProcessGroup
  //   distributed::ProcessGroup* pg = map->get(rid);
  //   std::vector<phi::DenseTensor> in_tensor;
  //   std::vector<phi::DenseTensor> out_tensor;
  //   in_tensor.push_back(*x);
  //   out_tensor.push_back(temp_out);
  //   auto task = pg->AllGather(in_tensor, out_tensor);
  //   task->Wait();
  // } else {
  //   auto comm = platform::NCCLCommContext::Instance().Get(rid, place);
  //   PADDLE_ENFORCE_EQ(
  //       nranks,
  //       comm->nranks(),
  //       platform::errors::InvalidArgument(
  //           "nranks: %s should equal to %s", nranks, comm->nranks()));

  //   int64_t send_numel = x->numel();
  //   const T* send_buff = x->data<T>();
  //   T* recv_buff = temp_out.data<T>();
  //   gpuStream_t stream = nullptr;
  // should ExecutionContext for calc stream.
  //   stream = ctx.cuda_device_context().stream();

  //   PADDLE_ENFORCE_GPU_SUCCESS(
  //       platform::dynload::ncclAllGather(send_buff,
  //                                        recv_buff,
  //                                        send_numel,
  //                                        static_cast<ncclDataType_t>(dtype),
  //                                        comm->comm(),
  //                                        stream));
  // }

  // std::vector<phi::DenseTensor> inputs;
  // int axis = x.dims().size() - 1;
  // auto out_dims = x.dims();
  // out_dims[out_dims.size() - 1] *= nranks;
  // int rows_per_tensor = x.dims()[0];
  // int offset = 0;
  // for (int i = 0; i < nranks; i++) {
  //   phi::DenseTensor temp = temp_out.Slice(offset, offset + rows_per_tensor);
  //   inputs.emplace_back(temp);
  //   offset += rows_per_tensor;
  // }

  // math::ConcatFunctor<phi::GPUContext, T> functor;
  // out->mutable_data<T>(out_dims, place);
  // auto& dev_ctx2 = ctx.template device_context<phi::GPUContext>();
  // functor(dev_ctx2, inputs, axis, out);
#else
  PADDLE_THROW(
      errors::PreconditionNotMet("PaddlePaddle should compile with GPU."));
#endif
}
}  // namespace phi

PD_REGISTER_KERNEL(concat_new,
                   GPU,
                   ALL_LAYOUT,
                   phi::ConcatNewKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
