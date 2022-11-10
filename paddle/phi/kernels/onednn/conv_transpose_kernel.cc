// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/phi/kernels/conv_transpose_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/data_layout_transform.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/phi/backends/onednn/onednn_helper.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/expect.h"
#include "paddle/phi/kernels/cpu/conv_util.h"

namespace phi {

inline dnnl::memory::dims GetWeightsTz(const phi::DenseTensor* filter,
                                       const int groups) {
  auto weights_tz = phi::vectorize(filter->dims());
  int g = std::max(groups, 1);
  int g_dim = (g > 1) ? 1 : 0;
  funcs::GetGroupConvWeightsTz(weights_tz, g);
  // gIOHW -> gOIHW || IOHW -> OIHW
  std::swap(weights_tz[g_dim + 0], weights_tz[g_dim + 1]);
  return weights_tz;
}

template <typename T, typename K, typename T_out>
class ConvTransposeOneDNNHandlerT
    : public funcs::OneDNNHandlerNoCachingT<T, dnnl::deconvolution_forward> {
 public:
  ConvTransposeOneDNNHandlerT(const OneDNNContext& dev_ctx,
                            const dnnl::engine engine,
                            Place cpu_place,
                            const phi::DenseTensor* input,
                            const phi::DenseTensor* filter,
                            const phi::DenseTensor* bias,
                            const std::vector<int>& strides_in,
                            const std::vector<int>& paddings_in,
                            const std::string& padding_algorithm,
                            const std::vector<int>& dilations_in,
                            int groups,
                            const std::string& data_format,
                            bool is_test,
                            bool is_BFLOAT16,
                            const std::string& fuse_activation,
                            bool force_fp32_output,
                            phi::DenseTensor* output)
      : funcs::OneDNNHandlerNoCachingT<T, dnnl::deconvolution_forward>(
            engine, cpu_place) {
    PADDLE_ENFORCE_EQ(is_test,
                      true,
                      phi::errors::InvalidArgument(
                          "ConvTransposeOneDNN works only for inference. "
                          "The attribute \'is_test\' value should be set to "
                          "True, but got is_test=False."));

    PADDLE_ENFORCE_EQ(
        input->layout(),
        DataLayout::ONEDNN,
        phi::errors::InvalidArgument(
            "Got wrong layout = %d for Input tensor.", input->layout()));

    PADDLE_ENFORCE_EQ(
        filter->layout(),
        DataLayout::ONEDNN,
        phi::errors::InvalidArgument(
            "The filter tensor's layout should be %d, but got %d.",
            DataLayout::ONEDNN,
            filter->layout()));

    PADDLE_ENFORCE_EQ(
        input->dims().size(),
        4,
        phi::errors::InvalidArgument("Input must be with 4 dimensions, "
                                          "i.e. NCHW. but got dimension =%d",
                                          input->dims().size()));
    PADDLE_ENFORCE_EQ(
        filter->dims().size(),
        4,
        phi::errors::InvalidArgument("Filter must be with 4 dimensions, "
                                          "i.e. OIHW, but got dimension =%d",
                                          filter->dims().size()));

    if (bias) {
      PADDLE_ENFORCE_EQ(
          bias->layout(),
          DataLayout::ONEDNN,
          phi::errors::InvalidArgument(
              "The bias tensor's laytout should be %d, but got %d.",
              DataLayout::ONEDNN,
              bias->layout()));

      PADDLE_ENFORCE_EQ(
          bias->dims().size(),
          1,
          phi::errors::InvalidArgument("Bias must only have 1 dimension, "
                                            "i.e. X, but got dimension = %d .",
                                            bias->dims().size()));
    }

    const auto input_dims = input->dims();
    const auto data_dims = phi::slice_ddim(input_dims, 2, input_dims.size());
    const auto filter_dims = filter->dims();
    const auto filter_data_dims =
        phi::slice_ddim(filter_dims, 2, filter_dims.size());
    const auto ksize = phi::vectorize(filter_data_dims);
    std::vector<int64_t> strides(begin(strides_in), end(strides_in));
    std::vector<int64_t> paddings(begin(paddings_in), end(paddings_in));
    std::vector<int64_t> dilations(begin(dilations_in), end(dilations_in));

    PADDLE_ENFORCE_EQ(
        strides.size(),
        2,
        phi::errors::Unimplemented(
            "Now we only support 2d oneDNN convolution transpose op"));

    UpdatePaddingAndDilation(
        &paddings, &dilations, padding_algorithm, data_dims, strides, ksize);

    std::transform(
        dilations.begin(), dilations.end(), dilations.begin(), [](int64_t i) {
          return i - 1;
        });

    const auto src_tz = phi::vectorize(input->dims());
    const auto weights_tz = GetWeightsTz(filter, groups);
    const auto dst_tz = phi::vectorize(output->dims());
    const auto onednn_paddings = funcs::ToOnednnPadding(paddings);

    /* create memory descriptor for convolution without specified format
     * ('any') which lets a primitive (convolution in this case) choose
     * the memory format preferred for best performance
     */
    auto chosen_memory_format = funcs::OneDNNMemoryFormat::any;
    auto data_type = dnnl::memory::data_type::f32;
    if (is_BFLOAT16 || std::is_same<T_out, dtype::bfloat16>::value) {
        data_type = dnnl::memory::data_type::bf16;
      }

    const auto src_md = funcs::OneDNNMemDesc(src_tz, data_type, chosen_memory_format);
    const auto weights_md = funcs::OneDNNMemDesc(
            weights_tz, data_type, funcs::OneDNNMemoryFormat::any);
    const auto dst_md = funcs::OneDNNMemDesc(
          dst_tz, funcs::OneDNNGetDataType<T_out>(), chosen_memory_format);

    const dnnl::primitive_attr conv_trans_attr = CreateConvAttrs(dev_ctx, fuse_activation); //!!!!
    auto fwd_prop_kind = is_test ? dnnl::prop_kind::forward_inference
                                  : dnnl::prop_kind::forward_training;
    if (bias) {
      std::vector<int64_t> bias_tz = phi::vectorize(bias->dims());
      const auto bias_md = funcs::OneDNNMemDesc(
              bias_tz, data_type, funcs::OneDNNMemoryFormat::x);
      this->AcquireForwardPrimitiveDescriptor(
          conv_trans_attr,
          fwd_prop_kind,
          dnnl::algorithm::deconvolution_direct,
          src_md,
          weights_md,
          bias_md,
          dst_md,
          strides,
          dilations,
          onednn_paddings[0],
          onednn_paddings[1]);
    } else {
      this->AcquireForwardPrimitiveDescriptor(
          conv_trans_attr,
          fwd_prop_kind,
          dnnl::algorithm::deconvolution_direct,
          src_md,
          weights_md,
          dst_md,
          strides,
          dilations,
          onednn_paddings[0],
          onednn_paddings[1]);
    }
  }

  dnnl::primitive_attr CreateConvAttrs(const OneDNNContext& dev_ctx, const std::string& fuse_activation) {
    dnnl::primitive_attr conv_attr;
    dnnl::post_ops post_operations;

    const float fuse_alpha = PADDLE_GET_CONST(
                      float, dev_ctx.GetDnnAttr("fuse_alpha"));
    const float fuse_beta = PADDLE_GET_CONST(
                      float, dev_ctx.GetDnnAttr("fuse_beta"));
    // Fusion with ReLU layer is executed through the PostOps feature. Create a
    // PostOps object and configure it to execute an eltwise relu operation.
    if (fuse_activation == "relu" || fuse_activation == "leaky_relu") {
      constexpr float scale = 1.0f;
      post_operations.append_eltwise(
          scale, dnnl::algorithm::eltwise_relu, fuse_alpha, fuse_beta);
    } else if (fuse_activation == "relu6") {
      constexpr float scale = 1.0f;
      post_operations.append_eltwise(
          scale, dnnl::algorithm::eltwise_bounded_relu, fuse_alpha, fuse_beta);
    } else if (fuse_activation == "swish") {
      constexpr float scale = 1.0f;
      post_operations.append_eltwise(
          scale, dnnl::algorithm::eltwise_swish, fuse_alpha, fuse_beta);
    }
    conv_attr.set_post_ops(post_operations);
    return conv_attr;
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemoryWithReorder(
      const phi::DenseTensor* input) {
    const T* input_data = input->data<T>();
    return funcs::OneDNNHandlerNoCachingT<T, dnnl::deconvolution_forward>::
        AcquireMemoryWithReorder(input->mem_desc(),
                                 this->fwd_pd_->src_desc(),
                                 funcs::to_void_cast<T>(input_data));
  }

  std::shared_ptr<dnnl::memory> AcquireWeightsMemoryWithReorder(
      const OneDNNContext& dev_ctx,
      const std::string& key,
      const phi::DenseTensor* filter,
      const int& groups) {
    const K* filter_data = filter->data<K>();
    auto weights_tz = GetWeightsTz(filter, groups);
    int g = std::max(groups, 1);

    auto user_src_md = funcs::OneDNNMemDesc(
        weights_tz,
        funcs::OneDNNGetDataType<K>(),
        (g == 1) ? funcs::OneDNNMemoryFormat::iohw : funcs::OneDNNMemoryFormat::giohw);

    return this->template AcquireMemoryWithReorder<K>(
        dev_ctx,
        user_src_md,
        this->fwd_pd_->weights_desc(),
        funcs::to_void_cast<K>(filter_data),
        key,
        "@weights_mem_p");
  }

  template <typename F = T>
  std::shared_ptr<dnnl::memory> AcquireMemoryWithReorder(
      const OneDNNContext& dev_ctx,
      const dnnl::memory::desc& user_md,
      const dnnl::memory::desc& target_md,
      void* ptr,
      const std::string& key,
      const std::string& suffix,
      bool is_persistent = false,
      const std::vector<float>& scale_data = {1.0f},
      int mask = 0) {
    const auto target_key = key + suffix + "_target";
    const auto key_reorder_p = key + suffix + "reorder_p";
    const auto user_key = key + suffix + "_user";

    auto target_memory_p =
        std::static_pointer_cast<dnnl::memory>(dev_ctx.GetBlob(target_key));

    if (target_memory_p == nullptr) {
      auto user_memory_p =
          std::make_shared<dnnl::memory>(user_md, this->engine_, ptr);
      if (user_md != target_md) {
        target_memory_p =
            std::make_shared<dnnl::memory>(target_md, this->engine_);
        dnnl::reorder::primitive_desc reorder_pdesc;
        if (funcs::is_int8<T>()) {
          dnnl::primitive_attr attr;
          attr.set_output_scales(mask, scale_data);
          reorder_pdesc = dnnl::reorder::primitive_desc(
              *user_memory_p, *target_memory_p, attr);
        } else {
          reorder_pdesc =
              dnnl::reorder::primitive_desc(*user_memory_p, *target_memory_p);
        }
        auto reorder_p = std::make_shared<dnnl::reorder>(reorder_pdesc);
        dev_ctx.SetBlob(key_reorder_p, reorder_p);

        auto& astream = OneDNNContext::tls().get_stream();
        paddle::platform::RecordEvent record_reorder(
            "int_reorder",
            paddle::platform::TracerEventType::UserDefined,
            2,
            paddle::platform::EventRole::kUniqueOp);
        reorder_p->execute(
            astream,
            {{DNNL_ARG_FROM, *user_memory_p}, {DNNL_ARG_TO, *target_memory_p}});
        astream.wait();
      } else {
        target_memory_p = user_memory_p;
      }
      dev_ctx.SetBlob(user_key, user_memory_p);
      dev_ctx.SetBlob(target_key, target_memory_p);
    } else if (!is_persistent) {
      auto& astream = OneDNNContext::tls().get_stream();

      auto user_memory_p =
          std::static_pointer_cast<dnnl::memory>(dev_ctx.GetBlob(user_key));
      user_memory_p->set_data_handle(ptr);

      // TODO(jczaja): Here we detect if reorder is cached it means it is needed
      // need to change this to get rid of keys
      auto reorder_p = std::static_pointer_cast<dnnl::reorder>(
          dev_ctx.GetBlob(key_reorder_p));
      if (reorder_p != nullptr) {
        paddle::platform::RecordEvent record_reorder(
            "int_reorder",
            paddle::platform::TracerEventType::UserDefined,
            2,
            paddle::platform::EventRole::kUniqueOp);
        reorder_p->execute(
            astream,
            {{DNNL_ARG_FROM, *user_memory_p}, {DNNL_ARG_TO, *target_memory_p}});
        astream.wait();
      }
    }
    return target_memory_p;
  }

  std::shared_ptr<dnnl::memory> AcquireBiasMemoryWithReorder(
      const OneDNNContext& dev_ctx,
      const std::string& key,
      const phi::DenseTensor* bias) {
    const K* bias_data = bias->data<K>();
    auto user_bias_md =
        funcs::OneDNNMemDesc(phi::vectorize(bias->dims()),
                                funcs::OneDNNGetDataType<K>(),
                                funcs::OneDNNMemoryFormat::x);
    return this->AcquireMemoryWithReorder(dev_ctx,
                                          user_bias_md,
                                          this->fwd_pd_->bias_desc(),
                                          funcs::to_void_cast<K>(bias_data),
                                          key,
                                          "@bias_mem_p");
  }
};


static dnnl::memory::data_type GetDstType(bool is_bfloat16, 
                                          bool force_fp32_output) {
  auto dst_dt = dnnl::memory::data_type::f32;
  if (!force_fp32_output && is_bfloat16) {
    dst_dt = dnnl::memory::data_type::bf16;
  }
  return dst_dt;
}

template <typename T, typename T_out>
void ComputeFP32(const OneDNNContext& dev_ctx,
                 const DenseTensor* input,
                 const DenseTensor* filter,
                 const DenseTensor* bias,
                 const std::vector<int>& strides,
                 const std::vector<int>& paddings,
                 const std::string& padding_algorithm,
                 const std::vector<int>& dilations,
                 int groups,
                 const std::string& data_format,
                 bool is_test,
                 bool is_BFLOAT16,
                 const std::string& fuse_activation,
                 bool force_fp32_output,
                 DenseTensor* output) {
  const auto& onednn_engine = dev_ctx.GetEngine();

  ConvTransposeOneDNNHandlerT<T, float, T_out> handler(dev_ctx,
                                                          onednn_engine,
                                                          dev_ctx.GetPlace(),
                                                          input,
                                                          filter,
                                                          bias,
                                                          strides,
                                                          paddings,
                                                          padding_algorithm,
                                                          dilations,
                                                          groups,
                                                          data_format,
                                                          is_test,
                                                          is_BFLOAT16,
                                                          fuse_activation,
                                                          force_fp32_output,
                                                          output);
    auto src_memory_p = handler.AcquireSrcMemoryWithReorder(input);
    // Caching Key for weights is needed
    std::string key = funcs::CreateKey(dev_ctx,
                                          dev_ctx.GetInputsName("Input")[0],
                                          dev_ctx.GetInputsName("Filter")[0],
                                          (bias ? dev_ctx.GetInputsName("Bias")[0] : ""));
    key = funcs::ExtendKeyWithThreadInfoIfNeeded(dev_ctx, key);
    auto weights_memory_p = handler.AcquireWeightsMemoryWithReorder(
        dev_ctx, key, filter, groups);

    std::shared_ptr<dnnl::memory> dst_memory_p =
        handler.template AcquireDstMemory<T_out>(output);
    auto conv_p = handler.AcquireForwardPrimitive();

    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC, *src_memory_p},
        {DNNL_ARG_WEIGHTS, *weights_memory_p},
        {DNNL_ARG_DST, *dst_memory_p}};

    if (bias) {
      auto bias_memory_p =
          handler.AcquireBiasMemoryWithReorder(dev_ctx, key, bias);
      args.insert({DNNL_ARG_BIAS, *bias_memory_p});
    }
    auto& astream = OneDNNContext::tls().get_stream();
    conv_p->execute(astream, args);
    astream.wait();
    output->set_mem_desc(dst_memory_p->get_desc());
  }

template <typename T, typename Context>
void Conv2dTransposeKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& filter,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::vector<int>& output_padding,
                           const IntArray& output_size,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilations,
                           const std::string& data_format,
                           DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      dev_ctx.GetPlace().GetType(),
      AllocationType::CPU,
      phi::errors::PreconditionNotMet("Operator DNNL Conv must use CPUPlace"));

  bool is_test = dev_ctx.HasDnnAttr("is_test")
                     ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("is_test"))
                     : false;
  bool is_BFLOAT16 =
      dev_ctx.HasDnnAttr("mkldnn_data_type")
          ? PADDLE_GET_CONST(std::string,
                             dev_ctx.GetDnnAttr("mkldnn_data_type")) ==
                "bfloat16"
          : false;
  const auto* bias =
      dev_ctx.HasDnnInput("Bias") ? dev_ctx.GetDnnInput("Bias") : nullptr;
  const std::string& fuse_activation =
      dev_ctx.HasDnnAttr("fuse_activation")
          ? PADDLE_GET_CONST(std::string, dev_ctx.GetDnnAttr("fuse_activation"))
          : "";
  bool force_fp32_output =
      dev_ctx.HasDnnAttr("force_fp32_output")
          ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("force_fp32_output"))
          : false;
  auto dst_dt = GetDstType(is_BFLOAT16,
                           force_fp32_output);

  if (dst_dt == dnnl::memory::data_type::f32) {
      ComputeFP32<T, float>(dev_ctx,
                            &x,
                            &filter,
                            bias,
                            strides,
                            paddings,
                            padding_algorithm,
                            dilations,
                            groups,
                            data_format,
                            is_test,
                            is_BFLOAT16,
                            fuse_activation,
                            force_fp32_output,
                            out);
    } else if (dst_dt == dnnl::memory::data_type::bf16) {
      ComputeFP32<T, dtype::bfloat16>(dev_ctx,
                                      &x,
                                      &filter,
                                      bias,
                                      strides,
                                      paddings,
                                      padding_algorithm,
                                      dilations,
                                      groups,
                                      data_format,
                                      is_test,
                                      is_BFLOAT16,
                                      fuse_activation,
                                      force_fp32_output,
                                      out);
    }
  }



}  // namespace phi

PD_REGISTER_KERNEL(conv2d_transpose,
                   OneDNN,
                   ONEDNN,
                   phi::Conv2dTransposeKernel,
                   float,
                   phi::dtype::bfloat16) {}
