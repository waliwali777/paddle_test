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

#pragma once

#include "paddle/pten/api/include/tensor.h"

#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/dense_tensor.h"

namespace paddle {
namespace experimental {

template <typename T>
class ScalarArrayList {
 public:
  // Constructor support implicit
  ScalarArrayList() = default;

  ScalarArrayList(const std::vector<int64_t>& vec) : array_(vec) {}  // NOLINT

  ScalarArrayList(const int64_t* date_value, int64_t n) {
    AssignData(date_value, n);
  }

  ScalarArrayList(const int32_t* date_value, int64_t n) {
    AssignData(date_value, n);
  }

  // The Tensor must have one dim
  ScalarArrayList(const T& tensor) {  // NOLINT
    size_t n = tensor.numel();
    array_.reserve(n);
    switch (tensor.type()) {
      case DataType::INT32:
        AssignData(tensor.template data<int32_t>(), n);
        break;
      case DataType::INT64:
        AssignData(tensor.template data<int64_t>(), n);
        break;
      default:
        PADDLE_THROW(paddle::platform::errors::InvalidArgument(
            "Data type error. Currently, The data type of ScalarArrayList "
            "only supports Tensor with int32 and int64, "
            "but now received %s.",
            tensor.type()));
    }
  }

  // The Tensor in vec must have only one element
  ScalarArrayList(const std::vector<T>& tensor_list) {  // NOLINT
    auto n = tensor_list.size();
    array_.reserve(n);
    if (!tensor_list.empty()) {
      DataType data_type = tensor_list[0].dtype();
      switch (data_type) {
        case DataType::INT32: {
          for (auto i = 0; i < n; i++) {
            PADDLE_ENFORCE_EQ(
                tensor_list[i].dtype(),
                data_type,
                paddle::platform::errors::InvalidArgument(
                    "The data_type of tensors in the list isn't consistent."
                    "the first tensor is %s, but %dth tensor is %s.",
                    data_type,
                    i,
                    tensor_list[i].data_type()));
            array_.push_back(*tensor_list[i].template data<int32_t>());
          }
          break;
        }
        case DataType::INT64: {
          for (auto i = 0; i < n; i++) {
            PADDLE_ENFORCE_EQ(
                tensor_list[i].dtype(),
                data_type,
                paddle::platform::errors::InvalidArgument(
                    "The data_type of tensors in the list isn't consistent."
                    "the first tensor is %s, but %dth tensor is %s.",
                    data_type,
                    i,
                    tensor_list[i].data_type()));
            array_.push_back(*tensor_list[i].template data<int64_t>());
          }
          break;
        }
        default:
          PADDLE_THROW(paddle::platform::errors::InvalidArgument(
              "Data type error. Currently, The data type of ScalarArrayList "
              "only supports Tensor with int32 and int64, "
              "but now received %s.",
              data_type));
      }
    }
  }

  template <typename TT>
  ScalarArrayList(const ScalarArrayList<TT>& other) : array_(other.GetData()) {}

  // template <typename TT>
  // ScalarArrayList(ScalarArrayList<TT>&& other) {
  //   swap(array_, other.array_);
  // }

  paddle::framework::DDim GetDim() const {
    return paddle::framework::make_ddim(array_);
  }

  const std::vector<int64_t>& GetData() const { return array_; }

 private:
  /// \brief Assign the data_ from const data pointer value of type T.
  template <typename TYPE>
  void AssignData(const TYPE* value_data, int64_t n) {
    if (value_data) {
      array_.reserve(n);
      for (auto i = 0; i < n; i++) {
        array_.push_back(static_cast<int64_t>(value_data[i]));
      }
    } else {
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "The input data pointer is null."));
    }
  }

 private:
  std::vector<int64_t> array_;
};

using ScalarArray =
    paddle::experimental::ScalarArrayList<paddle::experimental::Tensor>;

}  // namespace experimental
}  // namespace paddle

namespace pten {

using ScalarArray = paddle::experimental::ScalarArrayList<pten::DenseTensor>;
}
