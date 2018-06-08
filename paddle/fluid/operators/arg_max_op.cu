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

#include "paddle/fluid/operators/arg_max_op.h"

// REGISTER_ARG_MINMAX_KERNEL(arg_max, ArgMax, CUDA);

REGISTER_OP_CUDA_KERNEL(
    arg_max,
    paddle::operators::ArgMaxKernel<paddle::platform::CUDADeviceContext, float,
                                    int64_t>,
    paddle::operators::ArgMaxKernel<paddle::platform::CUDADeviceContext, double,
                                    int64_t>,
    paddle::operators::ArgMaxKernel<paddle::platform::CUDADeviceContext,
                                    int64_t, int64_t>,
    paddle::operators::ArgMaxKernel<paddle::platform::CUDADeviceContext,
                                    int32_t, int64_t>,
    paddle::operators::ArgMaxKernel<paddle::platform::CUDADeviceContext,
                                    int16_t, int64_t>,
    paddle::operators::ArgMaxKernel<paddle::platform::CUDADeviceContext, size_t,
                                    int64_t>,
    paddle::operators::ArgMaxKernel<paddle::platform::CUDADeviceContext,
                                    uint8_t, int64_t>);
