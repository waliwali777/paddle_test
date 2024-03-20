// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/transforms/onednn/conv_activation_mkldnn_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class ConvActivationFusePattern : public paddle::drr::DrrPatternBase {
 private:
  const size_t activation_count_;
  std::string activation_name_;
  /*
   * fused_level_ = 0 : conv2d + activation
    fused_level_ = 1 : conv2d + bias + activation
    fused_level_ = 2 : conv2d + residual + activation
    fused_level_ = 3 : conv2d + + bias + residual + activation
  */
  const int fused_level_;

 public:
  ConvActivationFusePattern(size_t activation_count,
                            const std::string &activation_name,
                            int fused_level)
      : activation_count_(activation_count),
        activation_name_(activation_name),
        fused_level_(fused_level) {}

  std::string name() const override {
    return "Conv" + std::to_string(fused_level_) + activation_name_ +
           "FusePattern";
  }

  uint32_t benefit() const override { return activation_count_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    std::string conv_name = paddle::dialect::Conv2dOp::name();
    if (fused_level_ > 0) {
      conv_name = paddle::onednn::dialect::FusedConv2dOp::name();
    }
    const auto &conv =
        pat.Op(conv_name,
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});

    std::string activation_name_op = "pd_op." + activation_name_;
    if (activation_name_ == "hard_swish") {
      // oneDNN use hard_swish, paddle use hardswish
      activation_name_op = "pd_op.hardswish";
    }
    const auto &activation = pat.Op(activation_name_op);
    if (fused_level_ == 1) {
      conv({&pat.Tensor("input"),
            &pat.Tensor("filter"),
            &pat.Tensor("bias"),
            &pat.Tensor("__@input_none_tensor@__")},
           {&pat.Tensor("conv2d_out")});

    } else if (fused_level_ == 2) {
      conv({&pat.Tensor("input"),
            &pat.Tensor("filter"),
            &pat.Tensor("__@input_none_tensor@__"),
            &pat.Tensor("residual_param")},
           {&pat.Tensor("conv2d_out")});

    } else if (fused_level_ == 3) {
      conv({&pat.Tensor("input"),
            &pat.Tensor("filter"),
            &pat.Tensor("bias"),
            &pat.Tensor("residual_param")},
           {&pat.Tensor("conv2d_out")});

    } else {
      conv({&pat.Tensor("input"), &pat.Tensor("filter")},
           {&pat.Tensor("conv2d_out")});
    }
    pat.Tensor("act_out") = activation(pat.Tensor("conv2d_out"));

    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      if (activation_name_ == "swish") {
        float beta = match_ctx.Attr<float>("beta");
        // x * sigmod(βx)
        if (beta < 0.0) {
          return false;
        }
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    auto fuse_residual = res.BoolAttr(false);
    if (fused_level_ > 1) {
      fuse_residual = res.BoolAttr(true);
    }
    auto fuse_beta = res.Float32Attr(0.0f);
    auto fuse_alpha = res.Float32Attr(0.0f);
    if (activation_name_ == "relu6") {
      fuse_beta = res.Float32Attr(6.0f);
    } else if (activation_name_ == "hard_swish") {
      // hard swish have not attr float threshold = 6.0f, float scale = 6.0f,
      // float offset = 3.0f attr But in previous implementation hard swish,
      // fuse_alpha=1.f / 6.f， fuse_beta=1.f / 2.f, it has fixed
      fuse_beta = res.Float32Attr(1.f / 2.f);
      fuse_alpha = res.Float32Attr(1.f / 6.f);
    } else if (activation_name_ == "swish") {
      fuse_alpha = res.StrAttr("swish");
    }

    const auto &fused_conv =
        res.Op(paddle::onednn::dialect::FusedConv2dOp::name(),
               {{
                   {"strides", pat.Attr("strides")},
                   {"paddings", pat.Attr("paddings")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"dilations", pat.Attr("dilations")},
                   {"groups", pat.Attr("groups")},
                   {"data_format", pat.Attr("data_format")},
                   {"mkldnn_data_type", res.StrAttr("float32")},
                   {"fuse_activation", res.StrAttr(activation_name_)},
                   {"fuse_residual_connection", fuse_residual},
                   {"force_fp32_output", res.BoolAttr(false)},
                   {"fuse_alpha", fuse_alpha},
                   {"fuse_beta", fuse_beta},
                   {"scale_in", res.Float32Attr(1.0f)},
                   {"scale_out", res.Float32Attr(1.0f)},
                   {"scale_in_eltwise", res.Float32Attr(1.0f)},
                   {"scale_weights", res.VectorFloatAttr({1.0f})},
               }});
    if (fused_level_ == 1) {
      fused_conv({&res.Tensor("input"),
                  &res.Tensor("filter"),
                  &res.Tensor("bias"),
                  &res.InputNoneTensor()},
                 {&res.Tensor("act_out")});

    } else if (fused_level_ == 2) {
      fused_conv({&res.Tensor("input"),
                  &res.Tensor("filter"),
                  &res.InputNoneTensor(),
                  &res.Tensor("residual_param")},
                 {&res.Tensor("act_out")});

    } else if (fused_level_ == 3) {
      fused_conv({&res.Tensor("input"),
                  &res.Tensor("filter"),
                  &res.Tensor("bias"),
                  &res.Tensor("residual_param")},
                 {&res.Tensor("act_out")});
    } else {
      fused_conv({&res.Tensor("input"),
                  &res.Tensor("filter"),
                  &res.InputNoneTensor(),
                  &res.InputNoneTensor()},
                 {&res.Tensor("act_out")});
    }
  }
};

class ConvLeakyReluFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string activation_name_;
  const int fused_level_;

 public:
  ConvLeakyReluFusePattern(const std::string &activation_name, int fused_level)
      : activation_name_(activation_name), fused_level_(fused_level) {}

  std::string name() const override {
    return "Conv" + std::to_string(fused_level_) + "LeakyReluFusePattern";
  }

  uint32_t benefit() const override { return fused_level_ + 1; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    std::string conv_name = paddle::dialect::Conv2dOp::name();
    if (fused_level_ > 0) {
      conv_name = paddle::onednn::dialect::FusedConv2dOp::name();
    }

    const auto &conv =
        pat.Op(conv_name,
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});

    const auto &activation = pat.Op(
        activation_name_, {{"negative_slope", pat.Attr("negative_slope")}});
    if (fused_level_ == 1) {
      conv({&pat.Tensor("input"),
            &pat.Tensor("filter"),
            &pat.Tensor("bias"),
            &pat.Tensor("__@input_none_tensor@__")},
           {&pat.Tensor("conv2d_out")});

    } else if (fused_level_ == 2) {
      conv({&pat.Tensor("input"),
            &pat.Tensor("filter"),
            &pat.Tensor("__@input_none_tensor@__"),
            &pat.Tensor("residual_param")},
           {&pat.Tensor("conv2d_out")});

    } else if (fused_level_ == 3) {
      conv({&pat.Tensor("input"),
            &pat.Tensor("filter"),
            &pat.Tensor("bias"),
            &pat.Tensor("residual_param")},
           {&pat.Tensor("conv2d_out")});

    } else {
      conv({&pat.Tensor("input"), &pat.Tensor("filter")},
           {&pat.Tensor("conv2d_out")});
    }
    pat.Tensor("act_out") = activation(pat.Tensor("conv2d_out"));

    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      float negative_slope = match_ctx.Attr<float>("negative_slope");
      // leaky relu alpha is a positive number
      if (negative_slope <= 0.0) {
        return false;
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    auto fuse_residual = res.BoolAttr(false);
    if (fused_level_ > 1) {
      fuse_residual = res.BoolAttr(true);
    }

    const auto &fused_conv =
        res.Op(paddle::onednn::dialect::FusedConv2dOp::name(),
               {{
                   {"strides", pat.Attr("strides")},
                   {"paddings", pat.Attr("paddings")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"dilations", pat.Attr("dilations")},
                   {"groups", pat.Attr("groups")},
                   {"data_format", pat.Attr("data_format")},
                   {"mkldnn_data_type", res.StrAttr("float32")},
                   {"fuse_activation", res.StrAttr("leaky_relu")},
                   {"fuse_residual_connection", fuse_residual},
                   {"force_fp32_output", res.BoolAttr(false)},
                   {"fuse_alpha", pat.Attr("negative_slope")},
                   {"fuse_beta", res.Float32Attr(0.0f)},
                   {"scale_in", res.Float32Attr(1.0f)},
                   {"scale_out", res.Float32Attr(1.0f)},
                   {"scale_in_eltwise", res.Float32Attr(1.0f)},
                   {"scale_weights", res.VectorFloatAttr({1.0f})},
               }});

    if (fused_level_ == 1) {
      fused_conv({&res.Tensor("input"),
                  &res.Tensor("filter"),
                  &res.Tensor("bias"),
                  &res.InputNoneTensor()},
                 {&res.Tensor("act_out")});

    } else if (fused_level_ == 2) {
      fused_conv({&res.Tensor("input"),
                  &res.Tensor("filter"),
                  &res.InputNoneTensor(),
                  &res.Tensor("residual_param")},
                 {&res.Tensor("act_out")});

    } else if (fused_level_ == 3) {
      fused_conv({&res.Tensor("input"),
                  &res.Tensor("filter"),
                  &res.Tensor("bias"),
                  &res.Tensor("residual_param")},
                 {&res.Tensor("act_out")});
    } else {
      fused_conv({&res.Tensor("input"),
                  &res.Tensor("filter"),
                  &res.InputNoneTensor(),
                  &res.InputNoneTensor()},
                 {&res.Tensor("act_out")});
    }
  }
};

class ConvHardSigmoidFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string activation_name_;
  const int fused_level_;

 public:
  ConvHardSigmoidFusePattern(const std::string &activation_name,
                             int fused_level)
      : activation_name_(activation_name), fused_level_(fused_level) {}

  std::string name() const override { return "ConvHardSigmoidFusePattern"; }

  uint32_t benefit() const override { return fused_level_ + 1; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    std::string conv_name = paddle::dialect::Conv2dOp::name();
    if (fused_level_ > 0) {
      conv_name = paddle::onednn::dialect::FusedConv2dOp::name();
    }

    const auto &conv =
        pat.Op(conv_name,
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});

    const auto &activation =
        pat.Op(activation_name_,
               {{"slope", pat.Attr("slope")}, {"offset", pat.Attr("offset")}});
    if (fused_level_ == 1) {
      conv({&pat.Tensor("input"),
            &pat.Tensor("filter"),
            &pat.Tensor("bias"),
            &pat.Tensor("__@input_none_tensor@__")},
           {&pat.Tensor("conv2d_out")});

    } else if (fused_level_ == 2) {
      conv({&pat.Tensor("input"),
            &pat.Tensor("filter"),
            &pat.Tensor("__@input_none_tensor@__"),
            &pat.Tensor("residual_param")},
           {&pat.Tensor("conv2d_out")});

    } else if (fused_level_ == 3) {
      conv({&pat.Tensor("input"),
            &pat.Tensor("filter"),
            &pat.Tensor("bias"),
            &pat.Tensor("residual_param")},
           {&pat.Tensor("conv2d_out")});

    } else {
      conv({&pat.Tensor("input"), &pat.Tensor("filter")},
           {&pat.Tensor("conv2d_out")});
    }
    pat.Tensor("act_out") = activation(pat.Tensor("conv2d_out"));

    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      // WIP: whether need conditions
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    auto fuse_residual = res.BoolAttr(false);
    if (fused_level_ > 1) {
      fuse_residual = res.BoolAttr(true);
    }

    const auto &fused_conv =
        res.Op(paddle::onednn::dialect::FusedConv2dOp::name(),
               {{
                   {"strides", pat.Attr("strides")},
                   {"paddings", pat.Attr("paddings")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"dilations", pat.Attr("dilations")},
                   {"groups", pat.Attr("groups")},
                   {"data_format", pat.Attr("data_format")},
                   {"mkldnn_data_type", res.StrAttr("float32")},
                   {"fuse_activation", res.StrAttr("hard_sigmoid")},
                   {"fuse_residual_connection", fuse_residual},
                   {"force_fp32_output", res.BoolAttr(false)},
                   {"fuse_alpha", pat.Attr("slope")},
                   {"fuse_beta", pat.Attr("offset")},
                   {"scale_in", res.Float32Attr(1.0f)},
                   {"scale_out", res.Float32Attr(1.0f)},
                   {"scale_in_eltwise", res.Float32Attr(1.0f)},
                   {"scale_weights", res.VectorFloatAttr({1.0f})},
               }});

    if (fused_level_ == 1) {
      fused_conv({&res.Tensor("input"),
                  &res.Tensor("filter"),
                  &res.Tensor("bias"),
                  &res.InputNoneTensor()},
                 {&res.Tensor("act_out")});

    } else if (fused_level_ == 2) {
      fused_conv({&res.Tensor("input"),
                  &res.Tensor("filter"),
                  &res.InputNoneTensor(),
                  &res.Tensor("residual_param")},
                 {&res.Tensor("act_out")});

    } else if (fused_level_ == 3) {
      fused_conv({&res.Tensor("input"),
                  &res.Tensor("filter"),
                  &res.Tensor("bias"),
                  &res.Tensor("residual_param")},
                 {&res.Tensor("act_out")});
    } else {
      fused_conv({&res.Tensor("input"),
                  &res.Tensor("filter"),
                  &res.InputNoneTensor(),
                  &res.InputNoneTensor()},
                 {&res.Tensor("act_out")});
    }
  }
};

class ConvGeluFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string activation_name_;
  const int fused_level_;

 public:
  ConvGeluFusePattern(const std::string &activation_name, int fused_level)
      : activation_name_(activation_name), fused_level_(fused_level) {}

  std::string name() const override { return "ConvGeluFusePattern"; }

  uint32_t benefit() const override { return 2; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    std::string conv_name = paddle::dialect::Conv2dOp::name();
    if (fused_level_ > 0) {
      conv_name = paddle::onednn::dialect::FusedConv2dOp::name();
    }

    const auto &conv =
        pat.Op(conv_name,
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});

    const auto &activation =
        pat.Op(activation_name_, {{"approximate", pat.Attr("approximate")}});
    if (fused_level_ == 1) {
      conv({&pat.Tensor("input"),
            &pat.Tensor("filter"),
            &pat.Tensor("bias"),
            &pat.Tensor("__@input_none_tensor@__")},
           {&pat.Tensor("conv2d_out")});

    } else if (fused_level_ == 2) {
      conv({&pat.Tensor("input"),
            &pat.Tensor("filter"),
            &pat.Tensor("__@input_none_tensor@__"),
            &pat.Tensor("residual_param")},
           {&pat.Tensor("conv2d_out")});

    } else if (fused_level_ == 3) {
      conv({&pat.Tensor("input"),
            &pat.Tensor("filter"),
            &pat.Tensor("bias"),
            &pat.Tensor("residual_param")},
           {&pat.Tensor("conv2d_out")});

    } else {
      conv({&pat.Tensor("input"), &pat.Tensor("filter")},
           {&pat.Tensor("conv2d_out")});
    }

    pat.Tensor("act_out") = activation(pat.Tensor("conv2d_out"));

    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      // WIP: whether need conditions
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &gelu = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> std::string {
          bool approximate = match_ctx.Attr<bool>("approximate");
          if (approximate) return "gelu_tanh";
          return "gelu_erf";
        });
    auto fuse_residual = res.BoolAttr(false);
    if (fused_level_ > 1) {
      fuse_residual = res.BoolAttr(true);
    }

    const auto &fused_conv =
        res.Op(paddle::onednn::dialect::FusedConv2dOp::name(),
               {{
                   {"strides", pat.Attr("strides")},
                   {"paddings", pat.Attr("paddings")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"dilations", pat.Attr("dilations")},
                   {"groups", pat.Attr("groups")},
                   {"data_format", pat.Attr("data_format")},
                   {"mkldnn_data_type", res.StrAttr("float32")},
                   {"fuse_activation", gelu},
                   {"fuse_residual_connection", fuse_residual},
                   {"force_fp32_output", res.BoolAttr(false)},
                   {"fuse_alpha", res.Float32Attr(0.0f)},
                   {"fuse_beta", res.Float32Attr(0.0f)},
                   {"scale_in", res.Float32Attr(1.0f)},
                   {"scale_out", res.Float32Attr(1.0f)},
                   {"scale_in_eltwise", res.Float32Attr(1.0f)},
                   {"scale_weights", res.VectorFloatAttr({1.0f})},
               }});

    if (fused_level_ == 1) {
      fused_conv({&res.Tensor("input"),
                  &res.Tensor("filter"),
                  &res.Tensor("bias"),
                  &res.InputNoneTensor()},
                 {&res.Tensor("act_out")});

    } else if (fused_level_ == 2) {
      fused_conv({&res.Tensor("input"),
                  &res.Tensor("filter"),
                  &res.InputNoneTensor(),
                  &res.Tensor("residual_param")},
                 {&res.Tensor("act_out")});

    } else if (fused_level_ == 3) {
      fused_conv({&res.Tensor("input"),
                  &res.Tensor("filter"),
                  &res.Tensor("bias"),
                  &res.Tensor("residual_param")},
                 {&res.Tensor("act_out")});
    } else {
      fused_conv({&res.Tensor("input"),
                  &res.Tensor("filter"),
                  &res.InputNoneTensor(),
                  &res.InputNoneTensor()},
                 {&res.Tensor("act_out")});
    }
  }
};

class ConvClipFusePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string activation_name_;
  const int fused_level_;

 public:
  ConvClipFusePattern(const std::string &activation_name, int fused_level)
      : activation_name_(activation_name), fused_level_(fused_level) {}

  std::string name() const override { return "ConvClipFusePattern"; }

  uint32_t benefit() const override { return 2; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    std::string conv_name = paddle::dialect::Conv2dOp::name();
    if (fused_level_ > 0) {
      conv_name = paddle::onednn::dialect::FusedConv2dOp::name();
    }

    const auto &full_1 = pat.Op(paddle::dialect::FullOp::name(),
                                {{"value", pat.Attr("full_1_value")}});
    const auto &full_2 = pat.Op(paddle::dialect::FullOp::name(),
                                {{"value", pat.Attr("full_2_value")}});
    pat.Tensor("min") = full_1();
    pat.Tensor("max") = full_2();
    const auto &conv =
        pat.Op(conv_name,
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});

    const auto &activation = pat.Op(activation_name_);
    if (fused_level_ == 1) {
      conv({&pat.Tensor("input"),
            &pat.Tensor("filter"),
            &pat.Tensor("bias"),
            &pat.Tensor("__@input_none_tensor@__")},
           {&pat.Tensor("conv2d_out")});

    } else if (fused_level_ == 2) {
      conv({&pat.Tensor("input"),
            &pat.Tensor("filter"),
            &pat.Tensor("__@input_none_tensor@__"),
            &pat.Tensor("residual_param")},
           {&pat.Tensor("conv2d_out")});

    } else if (fused_level_ == 3) {
      conv({&pat.Tensor("input"),
            &pat.Tensor("filter"),
            &pat.Tensor("bias"),
            &pat.Tensor("residual_param")},
           {&pat.Tensor("conv2d_out")});

    } else {
      conv({&pat.Tensor("input"), &pat.Tensor("filter")},
           {&pat.Tensor("conv2d_out")});
    }
    pat.Tensor("act_out") = activation(
        pat.Tensor("conv2d_out"), pat.Tensor("min"), pat.Tensor("max"));

    paddle::drr::ResultPattern res = pat.ResultPattern();
    auto fuse_residual = res.BoolAttr(false);
    if (fused_level_ > 1) {
      fuse_residual = res.BoolAttr(true);
    }

    const auto &fused_conv =
        res.Op(paddle::onednn::dialect::FusedConv2dOp::name(),
               {{
                   {"strides", pat.Attr("strides")},
                   {"paddings", pat.Attr("paddings")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"dilations", pat.Attr("dilations")},
                   {"groups", pat.Attr("groups")},
                   {"data_format", pat.Attr("data_format")},
                   {"mkldnn_data_type", res.StrAttr("float32")},
                   {"fuse_activation", res.StrAttr("clip")},
                   {"fuse_residual_connection", fuse_residual},
                   {"force_fp32_output", res.BoolAttr(false)},
                   {"fuse_alpha", pat.Attr("full_1_value")},
                   {"fuse_beta", pat.Attr("full_2_value")},
                   {"scale_in", res.Float32Attr(1.0f)},
                   {"scale_out", res.Float32Attr(1.0f)},
                   {"scale_in_eltwise", res.Float32Attr(1.0f)},
                   {"scale_weights", res.VectorFloatAttr({1.0f})},
               }});

    if (fused_level_ == 1) {
      fused_conv({&res.Tensor("input"),
                  &res.Tensor("filter"),
                  &res.Tensor("bias"),
                  &res.InputNoneTensor()},
                 {&res.Tensor("act_out")});

    } else if (fused_level_ == 2) {
      fused_conv({&res.Tensor("input"),
                  &res.Tensor("filter"),
                  &res.InputNoneTensor(),
                  &res.Tensor("residual_param")},
                 {&res.Tensor("act_out")});

    } else if (fused_level_ == 3) {
      fused_conv({&res.Tensor("input"),
                  &res.Tensor("filter"),
                  &res.Tensor("bias"),
                  &res.Tensor("residual_param")},
                 {&res.Tensor("act_out")});
    } else {
      fused_conv({&res.Tensor("input"),
                  &res.Tensor("filter"),
                  &res.InputNoneTensor(),
                  &res.InputNoneTensor()},
                 {&res.Tensor("act_out")});
    }
  }
};

class ConvActFusePass : public pir::PatternRewritePass {
 public:
  ConvActFusePass()
      : pir::PatternRewritePass("conv_activation_mkldnn_fuse_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);

    // This nine activations have no extra attribute, can use the same pattern
    std::vector<std::string> supported_activations_name = {"abs",
                                                           "sqrt",
                                                           "mish",
                                                           "relu",
                                                           "sigmoid",
                                                           "tanh",
                                                           "relu6",
                                                           "hard_swish",
                                                           "swish"};
    size_t pattern_num = 1;
    // conv + activation -> fused_conv2d
    for (auto activation : supported_activations_name) {
      ps.Add(paddle::drr::Create<ConvActivationFusePattern>(
          context, pattern_num, activation, 0));
      pattern_num++;
    }

    // conv + bias -> fused_conv2d + activation -> fused_conv2d
    for (auto activation : supported_activations_name) {
      ps.Add(paddle::drr::Create<ConvActivationFusePattern>(
          context, pattern_num, activation, 1));
      pattern_num++;
    }

    // conv + residual -> fused_conv2d + activation -> fused_conv2d
    for (auto activation : supported_activations_name) {
      ps.Add(paddle::drr::Create<ConvActivationFusePattern>(
          context, pattern_num, activation, 2));
      pattern_num++;
    }

    // conv + bias + residual -> fused_conv2d + activation -> fused_conv2d
    for (auto activation : supported_activations_name) {
      ps.Add(paddle::drr::Create<ConvActivationFusePattern>(
          context, pattern_num, activation, 3));
      pattern_num++;
    }

    ps.Add(paddle::drr::Create<ConvLeakyReluFusePattern>(
        context, paddle::dialect::LeakyReluOp::name(), 0));

    // conv + bias -> fused_conv2d + activation -> fused_conv2d
    ps.Add(paddle::drr::Create<ConvLeakyReluFusePattern>(
        context, paddle::dialect::LeakyReluOp::name(), 1));

    // conv + residual -> fused_conv2d + activation -> fused_conv2d
    ps.Add(paddle::drr::Create<ConvLeakyReluFusePattern>(
        context, paddle::dialect::LeakyReluOp::name(), 2));

    // conv + bias + residual -> fused_conv2d + activation -> fused_conv2d
    ps.Add(paddle::drr::Create<ConvLeakyReluFusePattern>(
        context, paddle::dialect::LeakyReluOp::name(), 3));

    ps.Add(paddle::drr::Create<ConvHardSigmoidFusePattern>(
        context, paddle::dialect::HardsigmoidOp::name(), 0));
    ps.Add(paddle::drr::Create<ConvHardSigmoidFusePattern>(
        context, paddle::dialect::HardsigmoidOp::name(), 1));
    ps.Add(paddle::drr::Create<ConvHardSigmoidFusePattern>(
        context, paddle::dialect::HardsigmoidOp::name(), 2));
    ps.Add(paddle::drr::Create<ConvHardSigmoidFusePattern>(
        context, paddle::dialect::HardsigmoidOp::name(), 3));

    ps.Add(paddle::drr::Create<ConvGeluFusePattern>(
        context, paddle::dialect::GeluOp::name(), 0));
    ps.Add(paddle::drr::Create<ConvGeluFusePattern>(
        context, paddle::dialect::GeluOp::name(), 1));
    ps.Add(paddle::drr::Create<ConvGeluFusePattern>(
        context, paddle::dialect::GeluOp::name(), 2));
    ps.Add(paddle::drr::Create<ConvGeluFusePattern>(
        context, paddle::dialect::GeluOp::name(), 3));

    ps.Add(paddle::drr::Create<ConvClipFusePattern>(
        context, paddle::dialect::ClipOp::name(), 0));
    ps.Add(paddle::drr::Create<ConvClipFusePattern>(
        context, paddle::dialect::ClipOp::name(), 1));
    ps.Add(paddle::drr::Create<ConvClipFusePattern>(
        context, paddle::dialect::ClipOp::name(), 2));
    ps.Add(paddle::drr::Create<ConvClipFusePattern>(
        context, paddle::dialect::ClipOp::name(), 3));

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateConv2dActFusePass() {
  /**1.
   *   conv
   *    |     ->  fused_conv
   * activation
   * 2.
   * fused_conv2d (bias or residual)
   *      |                         -> fused_conv2d
   *  activation
   */
  return std::make_unique<ConvActFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(conv_activation_mkldnn_fuse_pass, ConvActFusePass);
