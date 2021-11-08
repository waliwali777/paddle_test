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

#include <functional>
#include <memory>
#include <utility>

#include "paddle/pten/api/ext/ext_dll_decl.h"
#include "paddle/pten/api/ext/ext_dtype.h"
#include "paddle/pten/common/data_type.h"

namespace paddle {
namespace experimental {

class Tensor;

class AbstractAutogradMeta {
 public:
  // No AbstractAutogradMeta should be created
  virtual ~AbstractAutogradMeta() {}
};

/**
 * Tensor is the API description of the basic data structure in the
 * [ "Paddle Tensor Operation (pten)" Library ].
 *
 * It is not limited to a simple n-dimensional array.
 * It contains a smart pointer to `TensorImpl`. The data description contained
 * in Tensor is defined by TensorImpl. Tensor only defines the interface for
 * computation.
 *
 * This is a new Tensor design, which is independent of the original
 * framework::Tensor in fluid. The original Tensor will be gradually discarded
 * in the future.
 *
 * Note: Tensor can be NULL state, Tensor is meaningful only when the
 * TensorImpl to which it is pointed is not empty.
 *
 * Note: For the consistency of C++ API self, and the consistency between C++
 * API and Python API, all member methods of Tensor are named with lowercase
 * letters and underscores.
 *
 * Note: Tensor cannot be inherited. The heterogeneous Tensor implementation
 * can be achieved by inheriting the underlying TensorBase.
 *
 * Note: This Tensor API is suitable for training and custom operators,
 * another simple Tensor design may be required for inference.
 */

class Tensor final {
 public:
  /* Part 1: Construction and destruction methods */

  /**
   * @brief Construct a new Tensor object
   */
  Tensor() = default;

  /**
   * @brief Construct a new Tensor object by copy
   */
  Tensor(const Tensor&) = default;

  /**
   * @brief Construct a new Tensor object by move
   */
  Tensor(Tensor&&) = default;

  /**
   * @brief Construct a new Tensor object by a TensorBase pointer
   *
   * @param tensor_impl
   */
  explicit Tensor(std::shared_ptr<pten::TensorBase> tensor_impl);

  /**
   * @brief Construct a new Tensor object on the target place.
   * This is a deprecated method and may be removed in the future!
   *
   * @param place
   */
  explicit Tensor(const PlaceType& place);

  /**
   * @brief Construct a new Tensor object on the target place
   * with specified shape.
   * This is a deprecated method and may be removed in the future!
   *
   * @param place
   * @param shape
   */
  Tensor(const PlaceType& place, const std::vector<int64_t>& shape);

  /* Part 2: Dimension, DataType and DataLayout methods */

  /**
   * @brief Return the number of elements of Tensor.
   *
   * @return int64_t
   */
  int64_t numel() const;

  /**
   * @brief Get the size of current tensor.
   * The compatible method of `Tensor::numel()`.
   * This is a deprecated method and may be removed in the future!
   *
   * @return int64_t
   */
  int64_t size() const;

  /**
   * @brief Return the dimensions of Tensor.
   *
   * @return paddle::framework::DDim
   */
  paddle::framework::DDim dims() const;

  /**
   * @brief Return the shape (dimensions) of Tensor.
   * The compatible method of `Tensor::dims()`.
   * This is a deprecated method and may be removed in the future!
   *
   * @return std::vector<int64_t>
   */
  std::vector<int64_t> shape() const;

  /**
   * @brief Reset the shape of the tensor.
   * Reshape must be called before calling mutable_data() or
   * copy_to(const PlaceType& place).
   * This is a deprecated method and may be removed in the future!
   *
   * @param shape
   */
  void reshape(const std::vector<int64_t>& shape);

  /**
   * @brief Return the data type of Tensor.
   *
   * @return DataType
   */
  DataType dtype() const;

  /**
   * @brief Return the data type of Tensor.
   * The compatible method of `Tensor::dtype()`.
   * This is a deprecated method and may be removed in the future!
   *
   * @return DataType
   */
  DataType type() const;

  /**
   * @brief Return the layout of Tensor.
   *
   * @return DataLayout
   */
  DataLayout layout() const;

  /* Part 3: Device and Backend methods */

  /**
   * @brief Return the place (device) of Tensor.
   *
   * @return paddle::platform::Place
   */
  paddle::platform::Place place() const;

  /**
   * @brief Determine whether the tensor device is CPU
   *
   * @return true
   * @return false
   */
  bool is_cpu() const;

  /**
   * @brief Determine whether the tensor device is CPU
   *
   * @return true
   * @return false
   */
  bool is_cuda() const;

  /* Part 4: Data Access methods */

  /**
   * @brief Get the memory pointer in CPU or GPU with specific data type.
   * It's usually used to get the output data pointer.
   *
   * @tparam T
   * @return T*
   */
  template <typename T>
  T* mutable_data();

  /**
   * @brief Get the memory pointer in CPU or GPU with specific data type.
   * It's usually used to get the output data pointer.
   * This is a deprecated method and may be removed in the future!
   *
   * @tparam T
   * @param place
   * @return T*
   */
  template <typename T>
  T* mutable_data(const PlaceType& place);

  /**
   * @brief Get the const memory pointer directly.
   * It's usually used to get the output data pointer.
   *
   * @tparam T
   * @return T*
   */
  template <typename T>
  const T* data() const;

  /**
   * @brief Get the memory pointer directly.
   * It's usually used to get the output data pointer.
   * This is a deprecated method and may be removed in the future!
   *
   * @tparam T
   * @return T*
   */
  template <typename T>
  T* data() const;

  /**
   * @brief Return a sub-tensor of the given tensor.
   * It is usually used to extract a sub-tensor (which supports
   * modifying the data of the original tensor) to perform further
   * operations.
   *
   * @param begin_idx The index of the start row (inclusive) to slice.
   *                  The index number begins from 0.
   * @param end_idx The index of the end row (exclusive) to slice.
   *                 The index number begins from begin_idx + 1.
   * @return Tensor
   */
  Tensor slice(const int64_t begin_idx, const int64_t end_idx) const;

  /**
   * @brief Return the implemention of current Tensor.
   *
   * @return std::shared_ptr<pten::TensorBase>
   */
  std::shared_ptr<pten::TensorBase> impl() const;

  /**
   * @brief Set the implemention of current Tensor.
   *
   * @param impl
   */
  void set_impl(const std::shared_ptr<pten::TensorBase>& impl);

  // TODO(chenweihang): slice and split methods use kernels?

  /* Part 5: Data Transform methods */

  /**
   * @brief Copy the host memory to tensor data.
   * It's usually used to set the input tensor data.
   *
   * @tparam T
   * @param place of target place, of which the tensor will copy to.
   * @return Tensor
   */
  template <typename T>
  Tensor copy_to(const PlaceType& place) const;

  /**
   * @brief Cast datatype from one to another
   *
   * @param target_type
   * @return Tensor
   */
  Tensor cast(const DataType& target_type) const;

  /* Part 6: Status utils methods */

  /**
   * @brief Determine whether it is a meaningful Tensor
   *
   * @return true
   * @return false
   */
  bool defined() const { return impl_ != nullptr; }

  /**
   * @brief Determine whether Tensor is initialized
   *
   * @return true
   * @return false
   */
  bool initialized() const { return impl_->initialized(); }

  /**
   * @brief Reset the Tensor implementation
   */
  void reset() { impl_.reset(); }

  /* Part 7: Operator overloading */

  /**
   * @brief Assignment operator
   *
   * @param x
   * @return Tensor&
   */
  Tensor& operator=(const Tensor& x) & {
    impl_ = x.impl_;
    autograd_meta_ = x.autograd_meta_;
    return *this;
  }

  /**
   * @brief Move assignment operator
   *
   * @param x
   * @return Tensor&
   */
  Tensor& operator=(Tensor&& x) & {
    impl_ = std::move(x.impl_);
    autograd_meta_ = std::move(x.autograd_meta_);
    return *this;
  }

  /* Part 8: Autograd methods */

  /**
   * @brief Get the autograd meta object
   *
   * @return AbstractAutogradMeta*
   */
  AbstractAutogradMeta* get_autograd_meta() const {
    return autograd_meta_.get();
  }

  /**
   * @brief Set the autograd meta object
   *
   * @param autograd_meta
   */
  void set_autograd_meta(std::shared_ptr<AbstractAutogradMeta> autograd_meta) {
    autograd_meta_ = std::move(autograd_meta);
  }

  /* Part 9: Auto generated Tensor methods */

 private:
  /**
   * [ Why use abstract TensorImpl interface here? ]
   *
   * We hope that the data structure at the API level of the framework can be
   * unified to Tensor, but Tensor itself is heterogeneous.
   *
   * Tensor can generally be represented by void* and size_t, place.
   * This is suitable for most scenarios including CPU, CUDA, HIP, CPU, etc.,
   * but there are a few cases where this definition cannot be described,
   * such as the Tensor representation in third-party lib such as Metal,
   * OpenCL, etc., as well as some special Tensor implementations, including
   * Tensor containing only one Scalar value, or Tensor representing String,
   * etc.
   *
   * Therefore, we hope to use a unified interface to shield the underlying
   * heterogeneous Tensor implementation, so that the API level can be unified
   * to one `Tensor`.
   */
  std::shared_ptr<pten::TensorBase> impl_;

  /**
   * [ Why need abstract AbstractAutogradMeta here? ]
   *
   * Dynamic graphs need to hold backward information
   *
   * [ Why AutogradMeta not in TensorImpl? ]
   *
   * 1. AutogradMeta is only used in dynamic graph, It is execution-related
   *    information, not Tensor data description-related information.
   * 2. Kernel calculation does not require AutogradMeta.
   */
  std::shared_ptr<AbstractAutogradMeta> autograd_meta_{nullptr};

  /**
   * Tensor name: used for adapt original execution mechanism and debug analysis
   * in the development of new dygraph.
   */
  std::string name_;
};

}  // namespace experimental
}  // namespace paddle
