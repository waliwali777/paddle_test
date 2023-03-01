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

#include "paddle/phi/kernels/gaussian_kernel.h"

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/generator.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void GaussianKernel(const Context& ctx,
                    const IntArray& shape,
                    float mean,
                    float std,
                    int seed,
                    DataType dtype,
                    DenseTensor* out) {
  std::normal_distribution<float> dist(mean, std);
  int64_t size = out->numel();
  ctx.template Alloc<T>(out);
  auto* data = out->data();
  uint64_t seed_v = static_cast<uint64_t>(seed);
  // TODO(pangyoki): implement GetXPURandomEngine to set different seeds on
  // corresponding XPU device.
  std::shared_ptr<std::mt19937_64> engine;
  if (seed_v) {
    engine = std::make_shared<std::mt19937_64>();
    engine->seed(seed_v);
  } else {
    engine = ctx.GetGenerator()->GetCPUEngine();
  }

  std::unique_ptr<T[]> data_cpu(new T[size]);
  for (int64_t i = 0; i < size; ++i) {
    data_cpu[i] = dist(*engine);
  }
  paddle::memory::Copy(ctx.GetPlace(),
                       data,
                       phi::CPUPlace(),
                       reinterpret_cast<void*>(data_cpu.get()),
                       size * sizeof(T));
}

}  // namespace phi

PD_REGISTER_KERNEL(gaussian,
                   XPU,
                   ALL_LAYOUT,
                   phi::GaussianKernel,
                   float,
                   phi::dtype::float16) {}
