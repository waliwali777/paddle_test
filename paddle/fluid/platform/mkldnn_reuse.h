/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>
#include <vector>
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace platform {

using user_function = std::function<std::shared_ptr<float>(const float*)>;

class MKLDNNHandler {
 public:
  MKLDNNHandler(const MKLDNNDeviceContext& dev_ctx, mkldnn::engine engine,
                const std::string& base_key)
      : dev_ctx_(dev_ctx),
        engine_(engine),
        key_(base_key),
        is_reusing_(false) {}

  std::shared_ptr<mkldnn::memory> AcquireSrcMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_src_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireWeightsMemory(
      const mkldnn::memory::desc& md, void* ptr,
      user_function custom_func = {}) {
    return this->AcquireMemory(md, ptr, "@user_weights_mem_p", custom_func);
  }

  std::shared_ptr<mkldnn::memory> AcquireBiasMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_bias_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_dst_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffDstMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_diff_dst_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffSrcMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_diff_src_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireMemoryFromPrimitive(
      mkldnn::memory::primitive_desc mdp, void* ptr,
      const std::string& suffix) {
    auto local_key = key_ + suffix;
    auto mem_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    PADDLE_ENFORCE((mem_p != nullptr) || (is_reusing_ == false),
                   "Fail to find mem primitive in device context");
    if (mem_p == nullptr) {
      mem_p = std::make_shared<mkldnn::memory>(mdp, ptr);
      dev_ctx_.SetBlob(local_key, mem_p);
    } else {
      mem_p->set_data_handle(ptr);
      // Mark that reusing happenned. All primitives from operator instance
      // should be reused or none of them. So we check consistency
      is_reusing_ = true;
    }
    return mem_p;
  }

  // This incarnation of AcquireMemory can call user function eg. custom reorder
  // or preprocessing routine if needed
  std::shared_ptr<mkldnn::memory> AcquireMemory(
      const mkldnn::memory::desc& md, void* ptr, const std::string& suffix,
      user_function custom_func = {}) {
    /*Generate key*/
    auto local_key = key_ + suffix;
    auto mem_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    PADDLE_ENFORCE((mem_p != nullptr) || (is_reusing_ == false),
                   "Fail to find mem primitive in device context");
    if (mem_p == nullptr) {
      // Call custom reorder/preprocessing func if available
      if (custom_func) {
        auto reordered_data = custom_func(reinterpret_cast<const float*>(ptr));
        dev_ctx_.SetBlob(local_key + "-custom_reorder", reordered_data);
        ptr = reinterpret_cast<void*>(reordered_data.get());
      }

      mem_p = std::make_shared<mkldnn::memory>(
          mkldnn::memory::primitive_desc{md, engine_}, ptr);
      dev_ctx_.SetBlob(local_key, mem_p);
    } else {
      mem_p->set_data_handle(ptr);
      // Mark that reusing happenned. All primitives from operator instance
      // should be reused or none of them. So we check consistency
      is_reusing_ = true;
    }
    return mem_p;
  }

  std::shared_ptr<mkldnn::memory> AcquireMemory(
      const std::shared_ptr<mkldnn::memory>& user_memory_p,
      const std::shared_ptr<mkldnn::memory>& target_memory_p,
      const std::string& suffix,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto local_key = key_ + suffix;
    auto key_reorder_p = key_ + suffix + "reorder_p";

    auto stored_reorder_p = std::static_pointer_cast<mkldnn::reorder>(
        dev_ctx_.GetBlob(key_reorder_p));

    if (stored_reorder_p) {
      pipeline.push_back(*stored_reorder_p);
    } else {
      auto reorder_p =
          std::make_shared<mkldnn::reorder>(*user_memory_p, *target_memory_p);
      dev_ctx_.SetBlob(key_reorder_p, reorder_p);
      pipeline.push_back(*reorder_p);
    }

    return target_memory_p;
  }

  std::shared_ptr<mkldnn::memory> AcquireMemory(
      mkldnn::memory::primitive_desc& mpd,       // NOLINT
      mkldnn::memory::primitive_desc& user_mpd,  // NOLINT
      const std::shared_ptr<mkldnn::memory> user_memory_p,
      const std::string& suffix,
      std::vector<mkldnn::primitive>& pipeline,  // NOLINT
      bool is_persistent = false) {
    // create reorder primitive if the input format is not the preferred one
    auto local_key = key_ + suffix;
    auto key_reorder_p = key_ + suffix + "reorder_p";

    auto target_memory_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    PADDLE_ENFORCE((target_memory_p != nullptr) || (is_reusing_ == false),
                   "Fail to find mem primitive in device context");
    if (target_memory_p == nullptr) {
      target_memory_p = user_memory_p;
      std::shared_ptr<mkldnn::primitive> reorder_p;
      if (mpd != user_mpd) {
        target_memory_p = std::make_shared<mkldnn::memory>(mpd);
        auto reorder_p =
            std::make_shared<mkldnn::reorder>(*user_memory_p, *target_memory_p);
        dev_ctx_.SetBlob(key_reorder_p, reorder_p);
        pipeline.push_back(*reorder_p);
      }
      dev_ctx_.SetBlob(local_key, target_memory_p);
    } else if (!is_persistent) {
      // Make reorder if needed
      auto reorder_p = std::static_pointer_cast<mkldnn::reorder>(
          dev_ctx_.GetBlob(key_reorder_p));
      if (reorder_p != nullptr) {
        pipeline.push_back(*reorder_p);
      }
      is_reusing_ = true;
    }
    return target_memory_p;
  }

  static std::string GetHash(mkldnn::memory::dims& operand_dims,  // NOLINT
                             const std::string& suffix) {
    return dims2str(operand_dims) + suffix;
  }

 protected:
  static std::string dims2str(const mkldnn::memory::dims& operand_dims) {
    std::string dstr = "";
    for (size_t i = 0; i < operand_dims.size(); ++i) {
      dstr += std::to_string(operand_dims[i]) + "-";
    }
    return dstr;
  }

 protected:
  const MKLDNNDeviceContext& dev_ctx_;
  mkldnn::engine engine_;
  std::string key_;
  bool is_reusing_;
};

template <class forward_t, class backward_data_t, class backward_weights_t>
class ConvMKLDNNTemplateHandler : public MKLDNNHandler {
 public:
  ConvMKLDNNTemplateHandler(
      std::shared_ptr<typename forward_t::primitive_desc> conv_pd,
      const platform::MKLDNNDeviceContext& dev_ctx, mkldnn::engine engine,
      const std::string& base_key)
      : platform::MKLDNNHandler(dev_ctx, engine, base_key) {
    conv_pd_ = conv_pd;
  }

  ConvMKLDNNTemplateHandler(
      std::shared_ptr<typename forward_t::primitive_desc> conv_pd,
      std::shared_ptr<typename backward_data_t::primitive_desc>
          conv_bwd_data_pd,
      std::shared_ptr<typename backward_weights_t::primitive_desc>
          conv_bwd_weights_pd,
      const platform::MKLDNNDeviceContext& dev_ctx, mkldnn::engine engine,
      const std::string& base_key)
      : platform::MKLDNNHandler(dev_ctx, engine, base_key),
        conv_pd_(conv_pd),
        conv_bwd_weights_pd_(conv_bwd_weights_pd),
        conv_bwd_data_pd_(conv_bwd_data_pd) {
    // If we are in Grad operatgor then update a key with BWD suffix to
    // distinguish from FWD memory primitives
    key_ += "-BWD";
  }

  size_t GetDstMemorySize() const {
    return conv_pd_->dst_primitive_desc().get_size();
  }

  mkldnn::memory::format GetDstFormat() const {
    return static_cast<mkldnn::memory::format>(
        conv_pd_->dst_primitive_desc().desc().data.format);
  }

  size_t GetDiffWeightsMemorySize() const {
    return conv_bwd_weights_pd_->diff_weights_primitive_desc().get_size();
  }

  size_t GetDiffSourceMemorySize() const {
    return conv_bwd_data_pd_->diff_src_primitive_desc().get_size();
  }

  std::shared_ptr<mkldnn::memory> AcquireSrcMemoryFromWeightsPrimitive(
      const std::shared_ptr<mkldnn::memory> user_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto src_pd = conv_bwd_weights_pd_->src_primitive_desc();
    auto user_pd = user_memory_p->get_primitive_desc();
    return this->AcquireMemory(src_pd, user_pd, user_memory_p,
                               "@weights-src_mem_p", pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffDstMemoryFromWeightsPrimitive(
      const std::shared_ptr<mkldnn::memory> user_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto diff_dst_pd = conv_bwd_weights_pd_->diff_dst_primitive_desc();
    auto user_pd = user_memory_p->get_primitive_desc();
    return this->AcquireMemory(diff_dst_pd, user_pd, user_memory_p,
                               "@weights-diff_dst_mem_p", pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffWeightsMemoryFromWeightsPrimitive(
      void* ptr) {
    return this->AcquireMemoryFromPrimitive(
        conv_bwd_weights_pd_->diff_weights_primitive_desc(), ptr,
        "@diff_weights_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffDstMemoryFromDataPrimitive(
      const std::shared_ptr<mkldnn::memory> user_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto diff_dst_pd = conv_bwd_data_pd_->diff_dst_primitive_desc();
    auto user_pd = user_memory_p->get_primitive_desc();
    return this->AcquireMemory(diff_dst_pd, user_pd, user_memory_p,
                               "@data-diff_dst_mem_p", pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireWeightsMemoryFromDataPrimitive(
      const std::shared_ptr<mkldnn::memory> user_weights_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto weights_pd = conv_bwd_data_pd_->weights_primitive_desc();
    auto user_pd = user_weights_memory_p->get_primitive_desc();
    return this->AcquireMemory(weights_pd, user_pd, user_weights_memory_p,
                               "@data-weights_mem_p", pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireResidualDataMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_residual_data_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemoryFromResidualDataMemory(
      const std::shared_ptr<mkldnn::memory>& user_residual_memory_p,
      void* dst_ptr,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    return this->AcquireMemory(user_residual_memory_p,
                               this->AcquireDstMemoryFromPrimitive(dst_ptr),
                               "@residual_data_mem_p", pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffSrcMemoryFromDataPrimitive(
      void* ptr) {
    return this->AcquireMemoryFromPrimitive(
        conv_bwd_data_pd_->diff_src_primitive_desc(), ptr, "@diff_src_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemoryFromPrimitive(void* ptr) {
    return this->AcquireMemoryFromPrimitive(conv_pd_->dst_primitive_desc(), ptr,
                                            "@dst_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireSrcMemoryFromPrimitive(
      const std::shared_ptr<mkldnn::memory> user_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto src_pd = conv_pd_->src_primitive_desc();
    auto user_pd = user_memory_p->get_primitive_desc();
    return this->AcquireMemory(src_pd, user_pd, user_memory_p, "@src_mem_p",
                               pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireWeightsMemoryFromPrimitive(
      const std::shared_ptr<mkldnn::memory> user_weights_memory_p,
      std::vector<mkldnn::primitive>& pipeline,  // NOLINT
      bool is_persistent = false) {
    auto user_weights_pd = user_weights_memory_p->get_primitive_desc();
    auto weights_pd = conv_pd_->weights_primitive_desc();
    return this->AcquireMemory(weights_pd, user_weights_pd,
                               user_weights_memory_p, "@weights_mem_p",
                               pipeline, is_persistent);
  }

  std::shared_ptr<mkldnn::memory> AcquireBiasMemoryFromPrimitive(
      const std::shared_ptr<mkldnn::memory> user_bias_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto user_bias_pd = user_bias_memory_p->get_primitive_desc();
    auto bias_pd = conv_pd_->bias_primitive_desc();
    return this->AcquireMemory(bias_pd, user_bias_pd, user_bias_memory_p,
                               "@bias_mem_p", pipeline);
  }

  std::shared_ptr<forward_t> AcquireConvolution(
      std::shared_ptr<mkldnn::memory> src_memory_p,
      std::shared_ptr<mkldnn::memory> weights_memory_p,
      std::shared_ptr<mkldnn::memory> dst_memory_p) {
    auto prim_key = key_ + "@conv_p";
    auto conv_p =
        std::static_pointer_cast<forward_t>(dev_ctx_.GetBlob(prim_key));
    PADDLE_ENFORCE((conv_p != nullptr) || (is_reusing_ == false),
                   "Fail to find convolution primitive in device context");
    if (conv_p == nullptr) {
      conv_p = std::make_shared<forward_t>(*conv_pd_, *(src_memory_p),
                                           *(weights_memory_p.get()),
                                           *(dst_memory_p.get()));

      dev_ctx_.SetBlob(prim_key, conv_p);
    } else {
      is_reusing_ = true;
    }
    return conv_p;
  }

  std::shared_ptr<forward_t> AcquireConvolution(
      std::shared_ptr<mkldnn::memory> src_memory_p,
      std::shared_ptr<mkldnn::memory> weights_memory_p,
      std::shared_ptr<mkldnn::memory> bias_memory_p,
      std::shared_ptr<mkldnn::memory> dst_memory_p) {
    auto prim_key = key_ + "@conv_p";
    auto conv_p =
        std::static_pointer_cast<forward_t>(dev_ctx_.GetBlob(prim_key));
    PADDLE_ENFORCE((conv_p != nullptr) || (is_reusing_ == false),
                   "Fail to find convolution primitive in device context");
    if (conv_p == nullptr) {
      conv_p = std::make_shared<forward_t>(
          *conv_pd_, *(src_memory_p), *(weights_memory_p.get()),
          *(bias_memory_p.get()), *(dst_memory_p.get()));

      dev_ctx_.SetBlob(prim_key, conv_p);
    } else {
      is_reusing_ = true;
    }
    return conv_p;
  }

  std::shared_ptr<backward_weights_t> AcquireConvolutionBackwardWeights(
      std::shared_ptr<mkldnn::memory> src_memory_p,
      std::shared_ptr<mkldnn::memory> diff_dst_memory_p,
      std::shared_ptr<mkldnn::memory> diff_weights_memory_p) {
    auto prim_key = key_ + "@conv_bwd_weights_p";
    auto conv_bwd_weights_p = std::static_pointer_cast<backward_weights_t>(
        dev_ctx_.GetBlob(prim_key));
    PADDLE_ENFORCE(
        (conv_bwd_weights_p != nullptr) || (is_reusing_ == false),
        "Fail to find convolution bwd weights primitive in device context");
    if (conv_bwd_weights_p == nullptr) {
      // create backward conv primitive for weights
      conv_bwd_weights_p = std::make_shared<backward_weights_t>(
          *conv_bwd_weights_pd_, *src_memory_p, *diff_dst_memory_p,
          *diff_weights_memory_p);
      dev_ctx_.SetBlob(prim_key, conv_bwd_weights_p);
    } else {
      is_reusing_ = true;
    }
    return conv_bwd_weights_p;
  }

  std::shared_ptr<backward_data_t> AcquireConvolutionBackwardData(
      std::shared_ptr<mkldnn::memory> diff_dst_memory_p,
      std::shared_ptr<mkldnn::memory> weights_memory_p,
      std::shared_ptr<mkldnn::memory> diff_src_memory_p) {
    auto prim_key = key_ + "@conv_bwd_data_p";
    auto conv_bwd_data_p =
        std::static_pointer_cast<backward_data_t>(dev_ctx_.GetBlob(prim_key));
    PADDLE_ENFORCE(
        (conv_bwd_data_p != nullptr) || (is_reusing_ == false),
        "Fail to find convolution bwd data primitive in device context");
    if (conv_bwd_data_p == nullptr) {
      conv_bwd_data_p = std::make_shared<backward_data_t>(
          *conv_bwd_data_pd_, *diff_dst_memory_p, *weights_memory_p,
          *diff_src_memory_p);
      dev_ctx_.SetBlob(prim_key, conv_bwd_data_p);
    } else {
      is_reusing_ = true;
    }
    return conv_bwd_data_p;
  }

  // Generate keys for storing/retriving primitives for this operator
  // TODO(jczaja): Make hashing function more optimial
  static std::string GetHash(mkldnn::memory::dims& input_dims,    // NOLINT
                             mkldnn::memory::dims& weights_dims,  // NOLINT
                             std::vector<int>& strides,           // NOLINT
                             std::vector<int>& paddings,          // NOLINT
                             std::vector<int>& dilations,         // NOLINT
                             int groups, const std::string& suffix) {
    return dims2str(input_dims) + dims2str(weights_dims) + dims2str(strides) +
           dims2str(paddings) + dims2str(dilations) + std::to_string(groups) +
           suffix;
  }

 private:
  std::shared_ptr<typename forward_t::primitive_desc> conv_pd_;
  std::shared_ptr<typename backward_weights_t::primitive_desc>
      conv_bwd_weights_pd_;
  std::shared_ptr<typename backward_data_t::primitive_desc> conv_bwd_data_pd_;
};

using ConvMKLDNNHandler =
    ConvMKLDNNTemplateHandler<mkldnn::convolution_forward,
                              mkldnn::convolution_backward_data,
                              mkldnn::convolution_backward_weights>;

using ConvTransposeMKLDNNHandler =
    ConvMKLDNNTemplateHandler<mkldnn::deconvolution_forward,
                              mkldnn::deconvolution_backward_data,
                              mkldnn::deconvolution_backward_weights>;
}  // namespace platform
}  // namespace paddle
