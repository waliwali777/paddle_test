// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

/*
 * ATTENTION this header file can only include in .cc file.
 */

#pragma once
#include "paddle/fluid/lite/core/op_registry.h"

USE_LITE_KERNEL(feed, kHost, kAny, kAny, def);
USE_LITE_KERNEL(fetch, kHost, kAny, kAny, def);

#ifdef LITE_WITH_ARM
USE_LITE_KERNEL(fc, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(mul, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(scale, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(softmax, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(conv2d, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(depthwise_conv2d, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_add, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(split, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(dropout, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(concat, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(pool2d, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(relu, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(transpose, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(transpose2, kARM, kFloat, kNCHW, def);
#endif

#ifdef LITE_WITH_X86
USE_LITE_KERNEL(relu, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(mul, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(fc, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(scale, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(square, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_sub, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_add, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(softmax, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(dropout, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(concat, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(conv2d, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(depthwise_conv2d, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(pool2d, kX86, kFloat, kNCHW, def);
#endif

#ifdef LITE_WITH_CUDA
USE_LITE_KERNEL(mul, kCUDA, kFloat, kNCHW, def);
USE_LITE_KERNEL(io_copy, kCUDA, kAny, kAny, host_to_device);
USE_LITE_KERNEL(io_copy, kCUDA, kAny, kAny, device_to_host);
#endif
