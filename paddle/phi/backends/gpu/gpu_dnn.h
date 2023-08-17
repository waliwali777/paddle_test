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

#pragma once

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_MUSA)

#ifdef PADDLE_WITH_HIP
#include "paddle/phi/backends/gpu/rocm/miopen_desc.h"
#include "paddle/phi/backends/gpu/rocm/miopen_helper.h"
#elif defined(PADDLE_WITH_MUSA)

#else  // CUDA
#include "paddle/phi/backends/gpu/cuda/cudnn_desc.h"
#include "paddle/phi/backends/gpu/cuda/cudnn_helper.h"
#endif

#endif
