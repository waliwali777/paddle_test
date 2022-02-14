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

#include "paddle/pten/kernels/fill_any_kernel.h"
#include "paddle/pten/backends/cpu/cpu_context.h"
#include "paddle/pten/common/float16.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/impl/fill_any_kernel_impl.h"

PT_REGISTER_KERNEL(fill_any,
                   CPU,
                   ALL_LAYOUT,
                   pten::FillAnyKernel,
                   bool,
                   int,
                   int64_t,
                   pten::dtype::float16,
                   float,
                   double) {}
