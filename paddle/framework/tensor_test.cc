/*
  Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#include "paddle/framework/tensor.h"
#include <gtest/gtest.h>
#include <string>

TEST(Tensor, Dims) {
  using namespace paddle::framework;
  using namespace paddle::platform;
  Tensor tt;
  tt.Resize({2, 3, 4});
  DDim dims = tt.dims();
  ASSERT_EQ(arity(dims), 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(i + 2, dims[i]);
  }
}

TEST(Tensor, DataAssert) {
  paddle::framework::Tensor src_tensor;

  bool caught = false;
  try {
    src_tensor.data<double>();
  } catch (paddle::platform::EnforceNotMet err) {
    caught = true;
    std::string msg =
        "holder_ should not be null\nTenosr holds no memory. Call "
        "Tensor::mutable_data first.";
    const char* what = err.what();
    for (size_t i = 0; i < msg.length(); ++i) {
      ASSERT_EQ(what[i], msg[i]);
    }
  }
  ASSERT_TRUE(caught);
}

/* following tests are not available at present
   because Memory::Alloc() and Memory::Free() have not been ready.
*/
TEST(Tensor, MutableData) {
  using namespace paddle::framework;
  using namespace paddle::platform;
  {
    Tensor src_tensor;
    float* p1 = nullptr;
    float* p2 = nullptr;
    // initialization
    p1 = src_tensor.mutable_data<float>(make_ddim({1, 2, 3}), CPUPlace());
    EXPECT_NE(p1, nullptr);
    // set src_tensor a new dim with large size
    // momery is supposed to be re-allocated
    p2 = src_tensor.mutable_data<float>(make_ddim({3, 4}), CPUPlace());
    EXPECT_NE(p2, nullptr);
    EXPECT_NE(p1, p2);
    // set src_tensor a new dim with same size
    // momery block is supposed to be unchanged
    p1 = src_tensor.mutable_data<float>(make_ddim({2, 2, 3}), CPUPlace());
    EXPECT_EQ(p1, p2);
    // set src_tensor a new dim with smaller size
    // momery block is supposed to be unchanged
    p2 = src_tensor.mutable_data<float>(make_ddim({2, 2}), CPUPlace());
    EXPECT_EQ(p1, p2);
  }

#ifndef PADDLE_ONLY_CPU
  {
    Tensor src_tensor;
    float* p1 = nullptr;
    float* p2 = nullptr;
    // initialization
    p1 = src_tensor.mutable_data<float>(make_ddim({1, 2, 3}), GPUPlace());
    EXPECT_NE(p1, nullptr);
    // set src_tensor a new dim with large size
    // momery is supposed to be re-allocated
    p2 = src_tensor.mutable_data<float>(make_ddim({3, 4}), GPUPlace());
    EXPECT_NE(p2, nullptr);
    EXPECT_NE(p1, p2);
    // set src_tensor a new dim with same size
    // momery block is supposed to be unchanged
    p1 = src_tensor.mutable_data<float>(make_ddim({2, 2, 3}), GPUPlace());
    EXPECT_EQ(p1, p2);
    // set src_tensor a new dim with smaller size
    // momery block is supposed to be unchanged
    p2 = src_tensor.mutable_data<float>(make_ddim({2, 2}), GPUPlace());
    EXPECT_EQ(p1, p2);
  }
#endif
}

TEST(Tensor, ShareDataWith) {
  using namespace paddle::framework;
  using namespace paddle::platform;
  {
    Tensor src_tensor;
    Tensor dst_tensor;
    // Try to share data form uninitialized tensor
    bool caught = false;
    try {
      dst_tensor.ShareDataWith<float>(src_tensor);
    } catch (paddle::platform::EnforceNotMet err) {
      caught = true;
      std::string msg =
          "holder_ should not be null\nTenosr holds no memory. Call "
          "Tensor::mutable_data first.";
      const char* what = err.what();
      for (size_t i = 0; i < msg.length(); ++i) {
        ASSERT_EQ(what[i], msg[i]);
      }
    }
    ASSERT_TRUE(caught);

    src_tensor.mutable_data<int>(make_ddim({2, 3, 4}), CPUPlace());
    dst_tensor.ShareDataWith<int>(src_tensor);
    ASSERT_EQ(src_tensor.data<int>(), dst_tensor.data<int>());
  }

#ifndef PADDLE_ONLY_CPU
  {
    Tensor src_tensor;
    Tensor dst_tensor;
    src_tensor.mutable_data<int>(make_ddim({2, 3, 4}), GPUPlace());
    dst_tensor.ShareDataWith<int>(src_tensor);
    ASSERT_EQ(src_tensor.data<int>(), dst_tensor.data<int>());
  }
#endif
}

TEST(Tensor, Slice) {
  using namespace paddle::framework;
  using namespace paddle::platform;
  {
    Tensor src_tensor;
    src_tensor.mutable_data<int>(make_ddim({5, 3, 4}), CPUPlace());
    Tensor slice_tensor = src_tensor.Slice<int>(1, 3);
    DDim slice_dims = slice_tensor.dims();
    ASSERT_EQ(arity(slice_dims), 3);
    EXPECT_EQ(slice_dims[0], 2);
    EXPECT_EQ(slice_dims[1], 3);
    EXPECT_EQ(slice_dims[2], 4);

    uintptr_t src_data_address =
        reinterpret_cast<uintptr_t>(src_tensor.data<int>());
    uintptr_t src_mutable_data_address = reinterpret_cast<uintptr_t>(
        src_tensor.mutable_data<int>(src_tensor.dims(), CPUPlace()));
    uintptr_t slice_data_address =
        reinterpret_cast<uintptr_t>(slice_tensor.data<int>());
    uintptr_t slice_mutable_data_address = reinterpret_cast<uintptr_t>(
        slice_tensor.mutable_data<int>(slice_tensor.dims(), CPUPlace()));
    EXPECT_EQ(src_data_address, src_mutable_data_address);
    EXPECT_EQ(slice_data_address, slice_mutable_data_address);
    EXPECT_EQ(src_data_address + 3 * 4 * 1 * sizeof(int), slice_data_address);
  }

#ifndef PADDLE_ONLY_CPU
  {
    Tensor src_tensor;
    src_tensor.mutable_data<double>(make_ddim({6, 9}), GPUPlace());
    Tensor slice_tensor = src_tensor.Slice<double>(2, 6);
    DDim slice_dims = slice_tensor.dims();
    ASSERT_EQ(arity(slice_dims), 2);
    EXPECT_EQ(slice_dims[0], 4);
    EXPECT_EQ(slice_dims[1], 9);

    uintptr_t src_data_address =
        reinterpret_cast<uintptr_t>(src_tensor.data<double>());
    uintptr_t src_mutable_data_address = reinterpret_cast<uintptr_t>(
        src_tensor.mutable_data<double>(src_tensor.dims(), GPUPlace()));
    uintptr_t slice_data_address =
        reinterpret_cast<uintptr_t>(slice_tensor.data<double>());
    uintptr_t slice_mutable_data_address = reinterpret_cast<uintptr_t>(
        slice_tensor.mutable_data<double>(slice_tensor.dims(), GPUPlace()));
    EXPECT_EQ(src_data_address, src_mutable_data_address);
    EXPECT_EQ(slice_data_address, slice_mutable_data_address);
    EXPECT_EQ(src_data_address + 9 * 2 * sizeof(double), slice_data_address);
  }
#endif
}

TEST(Tensor, CopyFrom) {
  using namespace paddle::framework;
  using namespace paddle::platform;
  {
    Tensor src_tensor;
    Tensor dst_tensor;

    int* src_ptr = src_tensor.mutable_data<int>(make_ddim({3, 3}), CPUPlace());

    int arr[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    memcpy(src_ptr, arr, 9 * sizeof(int));

    auto cpu_place = new paddle::platform::CPUPlace();
    dst_tensor.CopyFrom<int>(src_tensor, *cpu_place);

    const int* dst_ptr = dst_tensor.data<int>();
    ASSERT_NE(src_ptr, dst_ptr);
    for (size_t i = 0; i < 9; ++i) {
      EXPECT_EQ(src_ptr[i], dst_ptr[i]);
    }

    Tensor slice_tensor = src_tensor.Slice<int>(1, 2);
    dst_tensor.CopyFrom<int>(slice_tensor, *cpu_place);
    const int* slice_ptr = slice_tensor.data<int>();
    dst_ptr = dst_tensor.data<int>();
    ASSERT_NE(dst_ptr, slice_ptr);
    for (size_t i = 0; i < 3; ++i) {
      EXPECT_EQ(dst_ptr[i], slice_ptr[i]);
    }
  }
#ifndef PADDLE_ONLY_CPU
  {
    Tensor src_tensor;
    Tensor gpu_tensor;
    Tensor dst_tensor;

    int* src_ptr = src_tensor.mutable_data<int>(make_ddim({3, 3}), CPUPlace());

    int arr[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    memcpy(src_ptr, arr, 9 * sizeof(int));

    // CPU Tensor to GPU Tensor
    auto gpu_place = new paddle::platform::GPUPlace(0);
    gpu_tensor.CopyFrom<int>(src_tensor, *gpu_place);

    // GPU Tensor to CPU Tensor
    auto cpu_place = new paddle::platform::CPUPlace();
    dst_tensor.CopyFrom<int>(gpu_tensor, *cpu_place);

    // Compare Tensors
    const int* dst_ptr = dst_tensor.data<int>();
    ASSERT_NE(src_ptr, dst_ptr);
    for (size_t i = 0; i < 9; ++i) {
      EXPECT_EQ(src_ptr[i], dst_ptr[i]);
    }

    Tensor slice_tensor = src_tensor.Slice<int>(1, 2);

    // CPU Slice Tensor to GPU Tensor
    gpu_tensor.CopyFrom<int>(slice_tensor, *gpu_place);

    // GPU Tensor to CPU Tensor
    dst_tensor.CopyFrom<int>(gpu_tensor, *cpu_place);

    // Compare Slice Tensors
    const int* slice_ptr = slice_tensor.data<int>();
    dst_ptr = dst_tensor.data<int>();
    ASSERT_NE(dst_ptr, slice_ptr);
    for (size_t i = 0; i < 3; ++i) {
      EXPECT_EQ(dst_ptr[i], slice_ptr[i]);
    }
  }
#endif
}
