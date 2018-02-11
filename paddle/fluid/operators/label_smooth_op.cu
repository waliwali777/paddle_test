/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/label_smooth_op.h"

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    label_smooth,
    ops::LabelSmoothKernel<paddle::platform::CUDADeviceContext, float>,
    ops::LabelSmoothKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    label_smooth_grad,
    ops::LabelSmoothGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::LabelSmoothGradKernel<paddle::platform::CUDADeviceContext, double>);
