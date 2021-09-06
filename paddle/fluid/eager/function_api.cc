// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/function_api.h"

#include "paddle/top/api/all.h"
#include "paddle/top/core/dense_tensor.h"

#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/memory/memcpy.h"

template<typename T>
static void add_kernel(const pt::DenseTensor& t0, const pt::DenseTensor& t1, pt::DenseTensor& out) {
    const T* t0_ptr = t0.data<T>();
    const T* t1_ptr = t1.data<T>();
    T* out_ptr = out.mutable_data<T>();
    for(int i = 0; i < t0.numel(); i++) {
        out_ptr[i] = t0_ptr[i] + t1_ptr[i];
    }   
}

namespace egr {

static std::shared_ptr<paddle::platform::Place> _expected_place(nullptr);

const paddle::platform::Place& GetExpectedPlace() {
    return *_expected_place.get();
}

void SetExpectedPlace(const paddle::platform::Place& place) {
    _expected_place = std::make_shared<paddle::platform::Place>(place);
}

template<typename DeviceContext>
static void ScaleDeviceDispatch(const pt::DenseTensor& dense_tensor, DeviceContext& dev_ctx, 
                                float scale, float bias, bool bias_after_scale, pt::DenseTensor* dense_out) {
    switch(dense_tensor.type()) {
        case pt::DataType::kFLOAT64: {
            pt::Scale<double>(dev_ctx, dense_tensor /* tensor */, scale /* scale */, bias/* bias */, bias_after_scale/* bias_after_scale */, dense_out/* out tensor */);
            break;
        }
        case pt::DataType::kFLOAT32: {
            pt::Scale<float>(dev_ctx, dense_tensor /* tensor */, scale /* scale */, bias/* bias */, bias_after_scale/* bias_after_scale */, dense_out/* out tensor */);
            break;
        }
        case pt::DataType::kINT64: {
            pt::Scale<int64_t>(dev_ctx, dense_tensor /* tensor */, scale /* scale */, bias/* bias */, bias_after_scale/* bias_after_scale */, dense_out/* out tensor */);
            break;
        }
        case pt::DataType::kINT32: {
            pt::Scale<int32_t>(dev_ctx, dense_tensor /* tensor */, scale /* scale */, bias/* bias */, bias_after_scale/* bias_after_scale */, dense_out/* out tensor */);
            break;
        }
        default: {
            PADDLE_THROW(paddle::platform::errors::Fatal("Unsupported data type"));
            break;
        }
    }
}

static void FillConstCPUFunctor(pt::DenseTensor* tensor_dense, double value) {
    switch(tensor_dense->type()) {
        case pt::DataType::kINT64: {
            int64_t* data_ptr = tensor_dense->mutable_data<int64_t>();
            for(int i = 0; i < tensor_dense->numel(); i++)
                data_ptr[i] = static_cast<int64_t>(value);
            break;
        }
        case pt::DataType::kINT32: {
            int32_t* data_ptr = tensor_dense->mutable_data<int32_t>();
            for(int i = 0; i < tensor_dense->numel(); i++)
                data_ptr[i] = static_cast<int32_t>(value);
            break;
        }
        case pt::DataType::kFLOAT64: {
            double* data_ptr = tensor_dense->mutable_data<double>();
            for(int i = 0; i < tensor_dense->numel(); i++)
                data_ptr[i] = static_cast<double>(value);
            break;
        }
        case pt::DataType::kFLOAT32: {
            float* data_ptr = tensor_dense->mutable_data<float>();
            for(int i = 0; i < tensor_dense->numel(); i++)
                data_ptr[i] = static_cast<float>(value);
            break;
        }
        default: {
            PADDLE_THROW(paddle::platform::errors::Fatal("Only supports tensor with fp32, fp64, int32, int64 datatypes for now"));
            break;
        }
    }
}

static void FillConstCUDAFunctor(pt::DenseTensor* tensor_dense, double value) {
    paddle::platform::DeviceContextPool& pool = paddle::platform::DeviceContextPool::Instance();
    auto* dev_ctx = dynamic_cast<paddle::platform::CUDADeviceContext*>(pool.Get(paddle::platform::CUDAPlace()));
    auto stream = dev_ctx->stream();

    switch(tensor_dense->type()) {
        case pt::DataType::kINT64: {
            std::vector<int64_t> host_data(tensor_dense->numel(), static_cast<int64_t>(value));
            int64_t* device_ptr = tensor_dense->mutable_data<int64_t>();
            paddle::memory::Copy(paddle::platform::CUDAPlace(), device_ptr, paddle::platform::CPUPlace(), host_data.data(),
                                 sizeof(int64_t)*tensor_dense->numel(), stream);
            break;
        }
        case pt::DataType::kINT32: {
            std::vector<int32_t> host_data(tensor_dense->numel(), static_cast<int32_t>(value));
            int32_t* device_ptr = tensor_dense->mutable_data<int32_t>();
            paddle::memory::Copy(paddle::platform::CUDAPlace(), device_ptr, paddle::platform::CPUPlace(), host_data.data(),
                                 sizeof(int32_t)*tensor_dense->numel(), stream);
            break;
        }
        case pt::DataType::kFLOAT64: {
            std::vector<double> host_data(tensor_dense->numel(), static_cast<double>(value));
            double* device_ptr = tensor_dense->mutable_data<double>();
            paddle::memory::Copy(paddle::platform::CUDAPlace(), device_ptr, paddle::platform::CPUPlace(), host_data.data(),
                                 sizeof(double)*tensor_dense->numel(), stream);
            break;
        }
        case pt::DataType::kFLOAT32: {
            std::vector<float> host_data(tensor_dense->numel(), static_cast<float>(value));
            float* device_ptr = tensor_dense->mutable_data<float>();
            paddle::memory::Copy(paddle::platform::CUDAPlace(), device_ptr, paddle::platform::CPUPlace(), host_data.data(),
                                 sizeof(float)*tensor_dense->numel(), stream);
            break;
        }
        default: {
            PADDLE_THROW(paddle::platform::errors::Fatal("Only supports tensor with fp32, fp64, int32, int64 datatypes for now"));
            break;
        }
    }
}


void ScaleAPI(const pt::Tensor& x, float scale, float bias,
              bool bias_after_scale, std::vector<pt::Tensor>& outs) {

    // Run Forward Function
    auto dense_tensor = std::dynamic_pointer_cast<pt::DenseTensor>(x.impl());
    
    if(outs.size() != 1)
        PADDLE_THROW(paddle::platform::errors::Fatal("ScaleAPI should only return 1 tensor"));
    
    // Init output tensor
    auto tensor_meta = std::make_unique<pt::TensorMeta>(dense_tensor->dims(), dense_tensor->backend(), 
          dense_tensor->type(), dense_tensor->layout());
    auto dense_out = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
    
    // Handle Device Context
    const paddle::platform::Place& expected_kernel_place = GetExpectedPlace();
    paddle::platform::DeviceContextPool& pool = paddle::platform::DeviceContextPool::Instance();

    if(expected_kernel_place == paddle::platform::CPUPlace()) {
        auto* dev_ctx = dynamic_cast<paddle::platform::CPUDeviceContext*>(pool.Get(expected_kernel_place));
        if(!dev_ctx)
            PADDLE_THROW(paddle::platform::errors::Fatal("Backend mismatch"));
        ScaleDeviceDispatch<paddle::platform::CPUDeviceContext>(*dense_tensor.get(), *dev_ctx, scale, bias, bias_after_scale, dense_out.get());
    
    } else if(expected_kernel_place == paddle::platform::CUDAPlace()) {
        auto* dev_ctx = dynamic_cast<paddle::platform::CUDADeviceContext*>(pool.Get(expected_kernel_place));
        if(!dev_ctx)
            PADDLE_THROW(paddle::platform::errors::Fatal("Backend mismatch"));
        ScaleDeviceDispatch<paddle::platform::CUDADeviceContext>(*dense_tensor.get(), *dev_ctx, scale, bias, bias_after_scale, dense_out.get());
    
    } else {
            PADDLE_THROW(paddle::platform::errors::Fatal("Only CPU and CUDA Backend are supported for now"));
    }
    
    outs[0].SetImpl(dense_out);
}

void FillConstAPI(double value, const pt::DDim& ddim, const pt::Backend& backend, 
                  const pt::DataType& dtype, const pt::DataLayout& layout,
                  pt::Tensor& target) {

    // Create new tensor->impl and fill it with 1.0
    // Fill 1.0
    std::shared_ptr<pt::DenseTensor> tensor_dense = nullptr;
    if(!target.defined() || !target.initialized()) {
        std::unique_ptr<pt::TensorMeta> tensor_meta = std::make_unique<pt::TensorMeta>(ddim, backend, dtype, layout);
        tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
        target.SetImpl(tensor_dense);

    } else {
        tensor_dense = std::dynamic_pointer_cast<pt::DenseTensor>(target.impl());
    }
    
    if(!tensor_dense)
        PADDLE_THROW(paddle::platform::errors::Fatal("FillConstAPI Only supports InputBuffer with DenseTensor for now."));
    
    switch(tensor_dense->backend()) {
        case pt::Backend::kCPU: {
            FillConstCPUFunctor(tensor_dense.get(), value);
            break;
        }
        case pt::Backend::kCUDA: {
            FillConstCUDAFunctor(tensor_dense.get(), value);
            break;
        }
        default: {
            PADDLE_THROW(paddle::platform::errors::Fatal("Only CPU and CUDA Backend are supported for now"));
        }
    }
}

void AccumulateTensorsAPI(pt::Tensor& t0, const pt::Tensor& t1) {
    // Accumulate to t0
    std::shared_ptr<pt::DenseTensor> t0_dense = std::dynamic_pointer_cast<pt::DenseTensor>(t0.impl());
    std::shared_ptr<pt::DenseTensor> t1_dense = std::dynamic_pointer_cast<pt::DenseTensor>(t1.impl());
    
    if(!t0_dense)
        PADDLE_THROW(paddle::platform::errors::Fatal("AccumulateTensorsAPI Only supports InputBuffer with DenseTensor for now."));
    if(t0_dense->backend() != pt::Backend::kCPU)
        PADDLE_THROW(paddle::platform::errors::Fatal("AccumulateTensorsAPI Only supports tensors with CPU backend for now."));
    if(!t0.initialized())
        PADDLE_THROW(paddle::platform::errors::Fatal("Tensors to accumulate has not been initialized"));
    if(!t1_dense)
        PADDLE_THROW(paddle::platform::errors::Fatal("AccumulateTensorsAPI Only supports InputBuffer with DenseTensor for now."));
    if(t1_dense->backend() != pt::Backend::kCPU)
        PADDLE_THROW(paddle::platform::errors::Fatal("AccumulateTensorsAPI Only supports tensors with CPU backend for now."));
    if(!t1.initialized())
        PADDLE_THROW(paddle::platform::errors::Fatal("Tensors to accumulate has not been initialized"));
    
    if(t1.type() != t0.type())
        PADDLE_THROW(paddle::platform::errors::Fatal("Unable to accumulate tensors with different dtype"));
    if(t1.numel() != t0.numel())
        PADDLE_THROW(paddle::platform::errors::Fatal("Unable to accumulate tensors with different sizes"));
    

    // TODO: Replace this with call to add_kernel_api
    switch(t0.type()) {
        case pt::DataType::kINT64: {
            add_kernel<int64_t>(*t0_dense.get(), *t1_dense.get(), *t0_dense.get());
            break;
        }
        case pt::DataType::kINT32: {
            add_kernel<int32_t>(*t0_dense.get(), *t1_dense.get(), *t0_dense.get());
            break;
        }
        case pt::DataType::kFLOAT64: {
            add_kernel<double>(*t0_dense.get(), *t1_dense.get(), *t0_dense.get());
            break;
        }
        case pt::DataType::kFLOAT32: {
            add_kernel<float>(*t0_dense.get(), *t1_dense.get(), *t0_dense.get());
            break;
        }
        default: {
            PADDLE_THROW(paddle::platform::errors::Fatal("Only supports tensor with fp32, fp64, int32, int64 datatypes for now"));
            break;
        }
    }
}

} // namespace egr
