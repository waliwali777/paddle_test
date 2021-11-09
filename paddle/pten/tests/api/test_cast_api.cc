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

#include <gtest/gtest.h>
#include <memory>

#include "paddle/pten/api/include/manipulation.h"

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"

PT_DECLARE_MODULE(ManipulationCPU);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PT_DECLARE_MODULE(ManipulationCUDA);
#endif

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

// TODO(chenweihang): Remove this test after the API is used in the dygraph
TEST(API, cast) {
  // 1. create tensor
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<pten::DenseTensor>(
      alloc,
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            framework::make_ddim({3, 4}),
                            pten::DataLayout::NCHW));
  auto* dense_x_data = dense_x->mutable_data<float>();

  for (int i = 0; i < dense_x->numel(); i++) {
    dense_x_data[i] = i;
  }

  paddle::experimental::Tensor x(dense_x);
  pten::DataType out_dtype = pten::DataType::FLOAT64;
  // 2. test API
  auto out = paddle::experimental::cast(x, out_dtype);

  // 3. check result
  std::vector<int> expect_shape = {3, 4};
  ASSERT_EQ(out.shape().size(), 2);
  ASSERT_EQ(out.shape()[0], expect_shape[0]);
  ASSERT_EQ(out.shape()[1], expect_shape[1]);
  ASSERT_EQ(out.numel(), 12);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), pten::DataType::FLOAT64);
  ASSERT_EQ(out.layout(), pten::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);
  auto dense_out = std::dynamic_pointer_cast<pten::DenseTensor>(out.impl());
  auto* dense_out_data = dense_out->data<double>();
  for (int i = 0; i < dense_x->numel(); i++) {
    ASSERT_NEAR(dense_out_data[i], static_cast<double>(dense_x_data[i]), 1e-6f);
  }
}
