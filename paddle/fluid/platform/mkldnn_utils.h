/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <mkldnn.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace paddle {
namespace platform {

// TODO(jczaja): Move This to mkldnn_reuse.h
static std::string pddims2str(const mkldnn::memory::dims& operand_dims) {
  std::string dstr = "";
  for (size_t i = 0; i < operand_dims.size(); ++i) {
    dstr += std::to_string(operand_dims[i]) + "-";
  }
  return dstr;
}

// TODO(jczaja): Move This to mkldnn_reuse.h
static std::string pdGetHash(
    const mkldnn::memory::dims& operand_dims,  // NOLINT
    const std::string& suffix) {
  return pddims2str(operand_dims) + suffix;
}

inline std::shared_ptr<mkldnn::memory::primitive_desc>
create_prim_desc_from_dims(
    const std::vector<int>& ltz, mkldnn::memory::format fmt,
    mkldnn::memory::data_type data_type = mkldnn::memory::data_type::f32) {
  // Make hash of PD
  auto key = std::string("PD-") +
             pdGetHash(ltz, std::to_string(static_cast<int>(fmt)) + "-" +
                                std::to_string(static_cast<int>(data_type)));
  // If there is PD registered then return reference to it
  auto& pool = platform::DeviceContextPool::Instance();
  auto place = paddle::platform::CPUPlace();
  auto dev_ctx = dynamic_cast<platform::MKLDNNDeviceContext*>(pool.Get(place));

  auto mpd = std::static_pointer_cast<mkldnn::memory::primitive_desc>(
      dev_ctx->GetBlob(key));
  // if there is no PD then create one
  if (mpd == nullptr) {
    mkldnn_memory_desc_t mem_fmt;

    mem_fmt.primitive_kind = mkldnn_memory;
    mem_fmt.ndims = ltz.size();
    for (unsigned int i = 0; i < ltz.size(); ++i) {
      mem_fmt.dims[i] = ltz[i];  // logical dimensions (nchw format,
                                 // regardless physical layout)
    }
    mem_fmt.data_type = static_cast<mkldnn_data_type_t>(data_type);
    mem_fmt.format = static_cast<mkldnn_memory_format_t>(fmt);

    unsigned int total_stride = 1;
    for (int i = ltz.size() - 1; i >= 0; --i) {
      mem_fmt.layout_desc.blocking.padding_dims[i] =
          ltz[i];  // logical dimensions (nchw format, regardless physical
                   // layout)
      mem_fmt.layout_desc.blocking.block_dims[i] = 1;
      mem_fmt.layout_desc.blocking.offset_padding_to_data[i] = 0;  // no offset
      mem_fmt.layout_desc.blocking.strides[0][i] = total_stride;
      mem_fmt.layout_desc.blocking.strides[1][i] = 1;
      total_stride *= ltz[i];
    }
    auto& cpu_engine = dev_ctx->GetEngine();
    mem_fmt.layout_desc.blocking.offset_padding = 0;  // no initial offset
    mpd = std::make_shared<mkldnn::memory::primitive_desc>(mem_fmt, cpu_engine);
    dev_ctx->SetBlob(key, mpd);
  }

  return mpd;
}

inline mkldnn::memory::primitive_desc create_prim_desc_from_format(
    const std::vector<int>& ltz, const mkldnn::memory::format format,
    const mkldnn::memory::data_type data_type) {
  auto md = mkldnn::memory::desc({ltz}, data_type, format);
  auto& pool = platform::DeviceContextPool::Instance();
  auto place = paddle::platform::CPUPlace();
  auto dev_ctx = dynamic_cast<platform::MKLDNNDeviceContext*>(pool.Get(place));
  PADDLE_ENFORCE_NOT_NULL(dev_ctx, "Could not get valid device");
  auto& cpu_engine = dev_ctx->GetEngine();
  return mkldnn::memory::primitive_desc(md, cpu_engine);
}

inline mkldnn::memory::format mkldnn_fmt(int rank) {
  static std::unordered_map<int, mkldnn::memory::format> dict{
      {5, mkldnn::memory::format::ncdhw},
      {4, mkldnn::memory::format::nchw},
      {3, mkldnn::memory::format::ncw},
      {2, mkldnn::memory::format::nc},
      {1, mkldnn::memory::format::x}};
  auto iter = dict.find(static_cast<int>(rank));
  if (iter != dict.end()) return iter->second;
  return mkldnn::memory::format::blocked;
}

}  // namespace platform
}  // namespace paddle
