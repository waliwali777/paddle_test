/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/math_function.h"
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/operators/math/math_function_impl.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {
namespace math {

using float16 = paddle::platform::float16;

template <>
void batched_gemm<platform::CPUDeviceContext, float16>(
    const platform::CPUDeviceContext& context, const CBLAS_TRANSPOSE transA,
    const CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
    const float16 alpha, const float16* A, const float16* B, const float16 beta,
    float16* C, const int batchCount, const int64_t strideA,
    const int64_t strideB) {
  PADDLE_THROW("float16 batched_gemm not supported on CPU");
}

#ifdef PADDLE_WITH_MKLML
// Use cblas_{s,d}gemm_batched if available: Run with 1 group of size batchSize.
template <>
void batched_gemm<platform::CPUDeviceContext, float>(
    const platform::CPUDeviceContext& context, const CBLAS_TRANSPOSE transA,
    const CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C, const int batchCount, const int64_t strideA,
    const int64_t strideB) {
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  auto a_array = std::vector<const float*>(batchCount);
  auto b_array = std::vector<const float*>(batchCount);
  auto c_array = std::vector<float*>(batchCount);
  for (int k = 0; k < batchCount; ++k) {
    a_array[k] = &A[k * strideA];
    b_array[k] = &B[k * strideB];
    c_array[k] = &C[k * M * N];
  }
  cblas_sgemm_batch(CblasRowMajor, &transA, &transB, &M, &N, &K, &alpha,
                    a_array.data(), &lda, b_array.data(), &ldb, &beta,
                    c_array.data(), &ldc, 1 /* group_count */, &batchCount);
}

template <>
void batched_gemm<platform::CPUDeviceContext, double>(
    const platform::CPUDeviceContext& context, const CBLAS_TRANSPOSE transA,
    const CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C, const int batchCount, const int64_t strideA,
    const int64_t strideB) {
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  auto a_array = std::vector<const double*>(batchCount);
  auto b_array = std::vector<const double*>(batchCount);
  auto c_array = std::vector<double*>(batchCount);
  for (int k = 0; k < batchCount; ++k) {
    a_array[k] = &A[k * strideA];
    b_array[k] = &B[k * strideB];
    c_array[k] = &C[k * M * N];
  }
  cblas_dgemm_batch(CblasRowMajor, &transA, &transB, &M, &N, &K, &alpha,
                    a_array.data(), &lda, b_array.data(), &ldb, &beta,
                    c_array.data(), &ldc, 1 /* group_count */, &batchCount);
}
#else
// The below is a naive but correct serial implementation that just loops
// over the batch dimension. This is a fallback for when the batched gemm
// functions of Intel MKL are not available. In the future, this computation
// should be parallelized.
template <>
void batched_gemm<platform::CPUDeviceContext, float>(
    const platform::CPUDeviceContext& context, const CBLAS_TRANSPOSE transA,
    const CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C, const int batchCount, const int64_t strideA,
    const int64_t strideB) {
  for (int k = 0; k < batchCount; ++k) {
    const float* Ak = &A[k * strideA];
    const float* Bk = &B[k * strideB];
    float* Ck = &C[k * M * N];
    Blas<platform::CPUDeviceContext>(context).GEMM(transA, transB, M, N, K,
                                                   alpha, Ak, Bk, beta, Ck);
  }
}

template <>
void batched_gemm<platform::CPUDeviceContext, double>(
    const platform::CPUDeviceContext& context, const CBLAS_TRANSPOSE transA,
    const CBLAS_TRANSPOSE transB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C, const int batchCount, const int64_t strideA,
    const int64_t strideB) {
  for (int k = 0; k < batchCount; ++k) {
    const double* Ak = &A[k * strideA];
    const double* Bk = &B[k * strideB];
    double* Ck = &C[k * M * N];
    Blas<platform::CPUDeviceContext>(context).GEMM(transA, transB, M, N, K,
                                                   alpha, Ak, Bk, beta, Ck);
  }
}
#endif

template <>
void gemv<platform::CPUDeviceContext, float>(
    const platform::CPUDeviceContext& context, const bool trans_a, const int M,
    const int N, const float alpha, const float* A, const float* B,
    const float beta, float* C) {
  CBLAS_TRANSPOSE transA = (trans_a == false) ? CblasNoTrans : CblasTrans;
  cblas_sgemv(CblasRowMajor, transA, M, N, alpha, A, N, B, 1, beta, C, 1);
}

template <>
void gemv<platform::CPUDeviceContext, double>(
    const platform::CPUDeviceContext& context, const bool trans_a, const int M,
    const int N, const double alpha, const double* A, const double* B,
    const double beta, double* C) {
  CBLAS_TRANSPOSE transA = (trans_a == false) ? CblasNoTrans : CblasTrans;
  cblas_dgemv(CblasRowMajor, transA, M, N, alpha, A, N, B, 1, beta, C, 1);
}

template <>
void axpy<platform::CPUDeviceContext, float>(
    const platform::CPUDeviceContext& context, const int n, const float alpha,
    const float* x, float* y) {
  cblas_saxpy(n, alpha, x, 1, y, 1);
}

template <>
void axpy<platform::CPUDeviceContext, double>(
    const platform::CPUDeviceContext& context, const int n, const double alpha,
    const double* x, double* y) {
  cblas_daxpy(n, alpha, x, 1, y, 1);
}

template struct SetConstant<platform::CPUDeviceContext, platform::float16>;
template struct SetConstant<platform::CPUDeviceContext, float>;
template struct SetConstant<platform::CPUDeviceContext, double>;
template struct SetConstant<platform::CPUDeviceContext, int>;
template struct SetConstant<platform::CPUDeviceContext, int64_t>;
template struct SetConstant<platform::CPUDeviceContext, bool>;

#define DEFINE_CPU_TRANS(RANK)                                             \
  template struct Transpose<platform::CPUDeviceContext, platform::float16, \
                            RANK>;                                         \
  template struct Transpose<platform::CPUDeviceContext, float, RANK>;      \
  template struct Transpose<platform::CPUDeviceContext, double, RANK>;     \
  template struct Transpose<platform::CPUDeviceContext, int, RANK>;        \
  template struct Transpose<platform::CPUDeviceContext, int64_t, RANK>;    \
  template struct Transpose<platform::CPUDeviceContext, bool, RANK>;

DEFINE_CPU_TRANS(1);
DEFINE_CPU_TRANS(2);
DEFINE_CPU_TRANS(3);
DEFINE_CPU_TRANS(4);
DEFINE_CPU_TRANS(5);
DEFINE_CPU_TRANS(6);

struct TensorSetConstantCPU {
  TensorSetConstantCPU(framework::Tensor* tensor, float value)
      : tensor_(tensor), value_(value) {}
  template <typename T>
  void operator()() const {
    auto cpu = platform::CPUPlace();
    auto* begin = tensor_->mutable_data<T>(cpu);
    std::fill(begin, begin + tensor_->numel(), static_cast<T>(value_));
  }
  framework::Tensor* tensor_;
  float value_;
};

template <>
void set_constant_with_place<platform::CPUPlace>(
    const platform::DeviceContext& context, framework::Tensor* tensor,
    float value) {
  framework::VisitDataType(framework::ToDataType(tensor->type()),
                           TensorSetConstantCPU(tensor, value));
}

template <>
void set_constant_with_place<platform::CUDAPinnedPlace>(
    const platform::DeviceContext& context, framework::Tensor* tensor,
    float value) {
  framework::VisitDataType(framework::ToDataType(tensor->type()),
                           TensorSetConstantCPU(tensor, value));
}

struct TensorSetConstantWithPlace : public boost::static_visitor<void> {
  TensorSetConstantWithPlace(const platform::DeviceContext& context,
                             framework::Tensor* tensor, float value)
      : context_(context), tensor_(tensor), value_(value) {}

  template <typename Place>
  void operator()(Place place) const {
    set_constant_with_place<Place>(context_, tensor_, value_);
  }

  const platform::DeviceContext& context_;
  framework::Tensor* tensor_;
  float value_;
};

void set_constant(const platform::DeviceContext& context,
                  framework::Tensor* tensor, float value) {
  TensorSetConstantWithPlace func(context, tensor, value);
#ifdef PADDLE_WITH_CUDA
  tensor->place().apply_visitor(func);
#else
  func(platform::CPUPlace());
#endif
}

template <typename T>
struct RowwiseAdd<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& vector, framework::Tensor* output) {
    auto in_dims = input.dims();
    auto size = input.numel() / in_dims[0];
    PADDLE_ENFORCE_EQ(vector.numel(), size);
    PADDLE_ENFORCE_EQ(output->dims(), in_dims);

    auto in = framework::EigenMatrix<T>::From(input);
    auto vec = framework::EigenVector<T>::Flatten(vector);
    auto out = framework::EigenMatrix<T>::From(*output);

    for (int64_t i = 0; i < in_dims[0]; ++i) {
      out.chip(i, 0) = in.chip(i, 0) + vec;
    }
  }
};

template struct RowwiseAdd<platform::CPUDeviceContext, float>;
template struct RowwiseAdd<platform::CPUDeviceContext, double>;

template struct ColwiseSum<platform::CPUDeviceContext, float>;
template struct ColwiseSum<platform::CPUDeviceContext, double>;
template struct ColwiseSum<platform::CPUDeviceContext, int>;
template struct ColwiseSum<platform::CPUDeviceContext, int64_t>;

template struct RowwiseSum<platform::CPUDeviceContext, float>;
template struct RowwiseSum<platform::CPUDeviceContext, double>;

template struct RowwiseMean<platform::CPUDeviceContext, float>;
template struct RowwiseMean<platform::CPUDeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
