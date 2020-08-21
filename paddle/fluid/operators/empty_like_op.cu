/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/empty_like_op.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    empty_like, ops::EmptyLikeKernel<plat::CUDADeviceContext, float>,
    ops::EmptyLikeKernel<plat::CUDADeviceContext, double>,
    ops::EmptyLikeKernel<plat::CUDADeviceContext, int64_t>,
    ops::EmptyLikeKernel<plat::CUDADeviceContext, int>,
    ops::EmptyLikeKernel<plat::CUDADeviceContext, plat::float16>);
