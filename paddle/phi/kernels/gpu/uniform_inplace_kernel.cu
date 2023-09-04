/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/uniform_inplace_kernel.h"

#include <thrust/random.h>

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/distribution_helper.h"
#include "paddle/phi/kernels/funcs/index_impl.cu.h"
#include "paddle/utils/flags.h"

namespace phi {

template <typename T>
struct UniformGenerator {
  T min_, max_;
  unsigned int seed_;
  T diag_val_;
  unsigned int diag_num_;
  unsigned int diag_step_;
  __host__ __device__ UniformGenerator(
      T min, T max, int seed, int diag_num, int diag_step, T diag_val)
      : min_(min),
        max_(max),
        seed_(seed),
        diag_num_(diag_num),
        diag_step_(diag_step),
        diag_val_(diag_val) {}

  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(seed_);
    thrust::uniform_real_distribution<T> dist(min_, max_);
    rng.discard(n);
    T out = dist(rng);
    unsigned int remainder = n % (diag_step_ + 1);
    if (remainder == 0 && diag_num_ > n / (diag_step_ + 1)) {
      out = diag_val_;
    }
    return out;
  }
};

template <typename T, typename Context>
void UniformInplaceKernel(const Context& ctx,
                          const DenseTensor& x,
                          float min,
                          float max,
                          int seed,
                          int diag_num,
                          int diag_step,
                          float diag_val,
                          DenseTensor* out) {
  ctx.template Alloc<T>(out);
  if (seed == 0) {
    // Use global Generator seed
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    funcs::uniform_distribution<MT> dist;
    funcs::uniform_real_transform<MT> trans(min, max);
    funcs::distribution_and_transform<T>(ctx, out, dist, trans);
  } else {
    // Use OP seed
    auto func = UniformGenerator<T>(static_cast<T>(min),
                                    static_cast<T>(max),
                                    seed,
                                    diag_num,
                                    diag_step,
                                    static_cast<T>(diag_val));
    IndexKernel<T, UniformGenerator<T>>(ctx, out, func);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(uniform_inplace,
                   GPU,
                   ALL_LAYOUT,
                   phi::UniformInplaceKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
