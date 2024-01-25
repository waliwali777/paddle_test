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

#include "paddle/fluid/pir/transforms/depthwise_conv_to_conv_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"

namespace {

class MapOp2AnotherPattern : public paddle::drr::DrrPatternBase {
 public:
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &depthwise_conv2d_op =
        pat.Op(paddle::dialect::DepthwiseConv2dOp::name(),
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});
    depthwise_conv2d_op({&pat.Tensor("input"), &pat.Tensor("filter")},
                        {&pat.Tensor("depthwise_conv2d_out")});
    pat.RequireNativeCall(
        [](const paddle::drr::MatchContext &match_ctx) -> bool {
#if CUDNN_VERSION >= 8100
          auto groups = match_ctx.Attr<int>("groups");
          if (groups > 1) {
            return true;
          } else {
            return false;
          }
#endif
          return false;
        });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &conv2d =
        pat.Op(paddle::dialect::Conv2dOp::name(),
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});
    conv2d({
        &res.Tensor("input"),
        &res.Tensor("filter"),
    } {&res.Tensor("depthwise_conv2d_out")});
  }
};

class DepthwiseConv2ConvPass : public pir::PatternRewritePass {
 public:
  DepthwiseConv2ConvPass()
      : pir::PatternRewritePass("depthwise_conv_to_conv_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(DepthWiseConv2d2Conv2dPattern().Build(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateDepthwiseConv2ConvPass() {
  return std::make_unique<DepthwiseConv2ConvPass>();
}
}  // namespace pir

REGISTER_IR_PASS(depthwise_conv_to_conv_pass, DepthwiseConv2ConvPass);
