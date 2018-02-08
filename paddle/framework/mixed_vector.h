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

#pragma once

#include <initializer_list>
#include <vector>

#include "paddle/framework/tensor.h"
#include "paddle/framework/tensor_util.h"

#include "glog/logging.h"

namespace paddle {
namespace framework {

template <typename T>
class Vector {
 public:
  using value_type = T;

  Vector() {
    size_ = 0;
    flag_ = kDataInCPU;
  }

  explicit Vector(size_t count, const T& value = T()) {
    resize(count);
    T* ptr = begin();
    for (size_t i = 0; i < count; ++i) {
      ptr[i] = value;
    }
  }

  Vector(std::initializer_list<T> init) {
    InitByIter(init.size(), init.begin(), init.end());
  }

  template <typename U>
  Vector(const std::vector<U>& dat) {  // NOLINT
    InitByIter(dat.size(), dat.begin(), dat.end());
  }

  Vector(const Vector<T>& other) { this->operator=(other); }

  Vector<T>& operator=(const Vector<T>& other) {
    if (other.size() != 0) {
      this->InitByIter(other.size(), other.begin(), other.end());
    } else {
      size_ = 0;
      flag_ = kDataInCPU;
    }
    return *this;
  }

  Vector(Vector<T>&& other) {
    this->size_ = other.size_;
    this->flag_ = other.flag_;
    if (other.cuda_vec_.capacity()) {
      this->cuda_vec_.ShareDataWith(other.cuda_vec_);
    }
    if (other.cpu_vec_.capacity()) {
      this->cpu_vec_.ShareDataWith(other.cpu_vec_);
    }
  }

  T& operator[](size_t i) {
    MutableCPU();
    return const_cast<T*>(cpu_vec_.data<T>())[i];
  }

  const T& operator[](size_t i) const {
    ImmutableCPU();
    return cpu_vec_.data<T>()[i];
  }

  size_t size() const { return size_; }

  T* begin() { return &this->operator[](0); }

  T* end() { return &this->operator[](size()); }

  T& front() { return *begin(); }

  T& back() {
    auto it = end();
    --it;
    return *it;
  }

  const T* begin() const { return &this->operator[](0); }
  const T* end() const { return &this->operator[](size()); }

  const T& back() const {
    auto it = end();
    --it;
    return *it;
  }

  const T& front() const { return *begin(); }

  template <typename Iter>
  void assign(Iter begin, Iter end) {
    InitByIter(end - begin, begin, end);
  }

  T* data() { return begin(); }

  const T* data() const { return begin(); }

  void push_back(T elem) {
    if (size_ + 1 > capacity()) {
      reserve((size_ + 1) << 1);
    }
    *end() = elem;
    ++size_;
  }

  void resize(size_t size) {
    if (size + 1 < capacity()) {
      size_ = size;
    } else {
      MutableCPU();
      Tensor cpu_tensor;
      platform::Place cpu = platform::CPUPlace();
      T* ptr = cpu_tensor.mutable_data<T>(
          framework::make_ddim({static_cast<int64_t>(size)}), cpu);
      const T* old_ptr =
          cpu_vec_.capacity() == 0 ? nullptr : cpu_vec_.data<T>();
      if (old_ptr != nullptr) {
        std::copy(old_ptr, old_ptr + size_, ptr);
      }
      size_ = size;
      cpu_vec_.ShareDataWith(cpu_tensor);
    }
  }

  const T* CUDAData(platform::Place place) const {
    PADDLE_ENFORCE(platform::is_gpu_place(place),
                   "CUDA Data must on CUDA place");
    ImmutableCUDA(place);
    return cuda_vec_.data<T>();
  }

  T* CUDAMutableData(platform::Place place) {
    const T* ptr = CUDAData(place);
    flag_ = kDirty | kDataInCUDA;
    return const_cast<T*>(ptr);
  }

  template <typename It>
  void Extend(It begin, It end) {
    size_t pre_size = size_;
    resize(pre_size + (end - begin));
    T* ptr = this->begin() + pre_size;
    for (; begin < end; ++begin, ++ptr) {
      *ptr = *begin;
    }
  }

  void clear() {
    size_ = 0;
    flag_ = kDirty | kDataInCPU;
  }

  size_t capacity() const {
    return cpu_vec_.capacity() / SizeOfType(typeid(T));
  }

  void reserve(size_t size) {
    size_t pre_size = size_;
    resize(size);
    resize(pre_size);
  }

  const T* Data(platform::Place place) const {
    if (platform::is_gpu_place(place)) {
      return CUDAData(place);
    } else {
      return data();
    }
  }

  T* MutableData(platform::Place place) {
    if (platform::is_gpu_place(place)) {
      return CUDAMutableData(place);
    } else {
      return data();
    }
  }

  operator std::vector<T>() const {
    std::vector<T> result;
    result.resize(size());
    std::copy(begin(), end(), result.begin());
    return result;
  }

  bool operator==(const Vector<T>& other) const {
    if (size() != other.size()) return false;
    for (auto it1 = begin(), it2 = other.begin(); it1 < end(); ++it1, ++it2) {
      if (*it1 != *it2) {
        return false;
      }
    }
    return true;
  }

 private:
  template <typename Iter>
  void InitByIter(size_t size, Iter begin, Iter end) {
    platform::Place cpu = platform::CPUPlace();
    T* ptr = this->cpu_vec_.template mutable_data<T>(
        framework::make_ddim({static_cast<int64_t>(size)}), cpu);
    for (size_t i = 0; i < size; ++i) {
      *ptr++ = *begin++;
    }
    flag_ = kDataInCPU | kDirty;
    size_ = size;
  }

  enum DataFlag { kDataInCPU = 0x01, kDataInCUDA = 0x02, kDirty = 0x10 };

  void MutableCPU() {
    if (IsInCUDA() && IsDirty()) {
      // COPY GPU Data To CPU
      Copy(cuda_vec_, platform::CPUPlace(), &cpu_vec_);
      WaitPlace(cuda_vec_.place());
    }
    flag_ = kDirty | kDataInCPU;
  }

  void ImmutableCUDA(platform::Place place) const {
    if (IsDirty()) {
      if (IsInCPU()) {
        Copy(cpu_vec_, boost::get<platform::CUDAPlace>(place), &cuda_vec_);
        WaitPlace(place);
        UnsetFlag(kDirty);
        SetFlag(kDataInCUDA);
      } else if (IsInCUDA() && !(place == cuda_vec_.place())) {
        framework::Tensor tmp;
        Copy(cuda_vec_, boost::get<platform::CUDAPlace>(place), &tmp);
        WaitPlace(cuda_vec_.place());
        cuda_vec_.ShareDataWith(tmp);
        // Still dirty
      } else {
        // Dirty && DataInCUDA && Device is same
        // Do nothing
      }
    } else {
      if (!IsInCUDA()) {
        // Even data is not dirty. However, data is not in CUDA. Copy data.
        Copy(cpu_vec_, boost::get<platform::CUDAPlace>(place), &cuda_vec_);
        WaitPlace(place);
        SetFlag(kDataInCUDA);
      } else if (!(place == cuda_vec_.place())) {
        framework::Tensor tmp;
        Copy(cuda_vec_, boost::get<platform::CUDAPlace>(place), &tmp);
        WaitPlace(cuda_vec_.place());
        cuda_vec_.ShareDataWith(tmp);
      } else {
        // Not Dirty && DataInCUDA && Device is same
        // Do nothing.
      }
    }
  }

  void ImmutableCPU() const {
    if (IsDirty() &&
        !IsInCPU()) {  // If data has been changed in CUDA, or CPU has no data.
      Copy(cuda_vec_, platform::CPUPlace(), &cpu_vec_);
      WaitPlace(cuda_vec_.place());
      UnsetFlag(kDirty);
    }
    SetFlag(kDataInCPU);
  }

  void UnsetFlag(int flag) const { flag_ &= ~flag; }
  void SetFlag(int flag) const { flag_ |= flag; }

  bool IsDirty() const { return flag_ & kDirty; }

  bool IsInCUDA() const { return flag_ & kDataInCUDA; }

  bool IsInCPU() const { return flag_ & kDataInCPU; }

  static void WaitPlace(const platform::Place place) {
    if (platform::is_gpu_place(place)) {
      platform::DeviceContextPool::Instance()
          .Get(boost::get<platform::CUDAPlace>(place))
          ->Wait();
    }
  }

  mutable int flag_;
  mutable Tensor cpu_vec_;
  mutable Tensor cuda_vec_;
  size_t size_;
};

}  // namespace framework
}  // namespace paddle
