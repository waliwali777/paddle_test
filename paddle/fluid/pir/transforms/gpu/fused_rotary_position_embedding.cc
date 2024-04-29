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

#include "paddle/fluid/pir/transforms/gpu/fused_rotary_position_embedding.h"

#include <string>

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class FusedRotaryPositionEmbeddingPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "RotaryPositionEmbeddingPattern"; }
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &squeeze = pat.Op(paddle::dialect::SqueezeOp::name());
    const auto &squeeze_1 = pat.Op(paddle::dialect::SqueezeOp::name());

    const auto &gather_nd = pat.Op(paddle::dialect::GatherNdOp::name());
    const auto &gather_nd_1 = pat.Op(paddle::dialect::GatherNdOp::name());
    const auto &unsqueeze = pat.Op(paddle::dialect::UnsqueezeOp::name());
    const auto &unsqueeze_1 = pat.Op(paddle::dialect::UnsqueezeOp::name());
    const auto &unsqueeze_2 = pat.Op(paddle::dialect::UnsqueezeOp::name());
    const auto &unsqueeze_4 = pat.Op(paddle::dialect::UnsqueezeOp::name());

    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &add_1 = pat.Op(paddle::dialect::AddOp::name());
    const auto &multiply1 = pat.Op(paddle::dialect::MultiplyOp::name());
    const auto &multiply2 = pat.Op(paddle::dialect::MultiplyOp::name());
    const auto &multiply3 = pat.Op(paddle::dialect::MultiplyOp::name());
    const auto &multiply4 = pat.Op(paddle::dialect::MultiplyOp::name());

    const auto &slice_q =
        pat.Op(paddle::dialect::SliceOp::name(),
               {{"axes", pat.Attr("axes_1")},
                {"decrease_axis", pat.Attr("decrease_axis_1")}});
    const auto &slice_q_1 =
        pat.Op(paddle::dialect::SliceOp::name(),
               {{"axes", pat.Attr("axes_2")},
                {"decrease_axis", pat.Attr("decrease_axis_2")}});

    const auto &slice_k =
        pat.Op(paddle::dialect::SliceOp::name(),
               {{"axes", pat.Attr("axes_1")},
                {"decrease_axis", pat.Attr("decrease_axis_1")}});

    const auto &slice_k_1 =
        pat.Op(paddle::dialect::SliceOp::name(),
               {{"axes", pat.Attr("axes_2")},
                {"decrease_axis", pat.Attr("decrease_axis_2")}});

    const auto &full_op = pat.Op(paddle::dialect::FullOp::name(),
                                 {{"shape", pat.Attr("shape")},
                                  {"value", pat.Attr("value")},
                                  {"dtype", pat.Attr("dtype")},
                                  {"place", pat.Attr("place")}});
    const auto &full_op_1 = pat.Op(paddle::dialect::FullOp::name(),
                                   {{"shape", pat.Attr("shape_1")},
                                    {"value", pat.Attr("full_op_1")},
                                    {"dtype", pat.Attr("dtype_1")},
                                    {"place", pat.Attr("place_1")}});
    const auto &full_op_2 = pat.Op(paddle::dialect::FullOp::name(),
                                   {{"shape", pat.Attr("shape_1")},
                                    {"value", pat.Attr("full_op_2")},
                                    {"dtype", pat.Attr("dtype_1")},
                                    {"place", pat.Attr("place_1")}});
    const auto &full_op_3 = pat.Op(paddle::dialect::FullOp::name(),
                                   {{"shape", pat.Attr("shape_1")},
                                    {"value", pat.Attr("full_op_3")},
                                    {"dtype", pat.Attr("dtype_1")},
                                    {"place", pat.Attr("place_1")}});
    // const auto &shape = pat.Op(paddle::dialect::ShapeOp::name());

    const auto &scale_op =
        pat.Op(paddle::dialect::ScaleOp::name(),
               {{"bias", pat.Attr("bias")},
                {"bias_after_scale", pat.Attr("bias_after_scale")}});

    const auto &scale_op_k =
        pat.Op(paddle::dialect::ScaleOp::name(),
               {{"bias", pat.Attr("bias_q")},
                {"bias_after_scale", pat.Attr("bias_after_scale_q")}});

    const auto &full_1 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("full_1_value")}});
    const auto &full_2 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("full_2_value")}});
    const auto &full_3 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("full_3_value")}});
    const auto &full_4 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("full_4_value")}});
    const auto &full_5 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("full_5_value")}});
    const auto &full_6 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("full_6_value")}});
    const auto &full_7 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("full_7_value")}});
    const auto &full_8 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("full_8_value")}});
    const auto &full_9 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                {{"value", pat.Attr("full_9_value")}});
    const auto &full_10 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                 {{"value", pat.Attr("full_10_value")}});
    const auto &full_11 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                 {{"value", pat.Attr("full_11_value")}});
    const auto &full_12 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                 {{"value", pat.Attr("full_12_value")}});
    const auto &full_13 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                 {{"value", pat.Attr("full_13_value")}});
    const auto &full_14 = pat.Op(paddle::dialect::FullIntArrayOp::name(),
                                 {{"value", pat.Attr("full_14_value")}});

    const auto &concat_op = pat.Op(paddle::dialect::ConcatOp::name());
    const auto &combine = pat.Op(pir::CombineOp::name());
    const auto &concat_op_k = pat.Op(paddle::dialect::ConcatOp::name());
    const auto &combine_k = pat.Op(pir::CombineOp::name());

    // squeeze对应的是150,151,147是输入,149是pd_op.full_int_array
    squeeze({&pat.Tensor("cos"), &full_13()},
            {&pat.Tensor("squeeze_out_cos"), &pat.Tensor("xshape")});

    // squeeze_1对应的是153,154,148是输入,152是pd_op.full_int_array
    squeeze_1({&pat.Tensor("sin"), &full_12()},
              {&pat.Tensor("squeeze_out_sin"), &pat.Tensor("xshape")});

    // unsqueeze对应的是156,157,17是position_ids,155是pd_op.full_int_array
    unsqueeze({&pat.Tensor("position_ids"), &full_11()},
              {&pat.Tensor("unsqueeze_s_out_cos"), &pat.Tensor("xshape")});

    // gather_nd对应的158,150是pd_op.squeeze,156对应的是pd_op.unsqueeze
    pat.Tensor("gather_nd_out_cos") = gather_nd(
        pat.Tensor("squeeze_out_cos"), pat.Tensor("unsqueeze_s_out_cos"));

    // unsqueeze_1对应的是160,161,158是pd_op_gather_nd,159是pd_op.full_int_array
    unsqueeze_1({&pat.Tensor("gather_nd_out_cos"), &full_10()},
                {&pat.Tensor("unsqueeze_out_cos"), &pat.Tensor("xshape")});

    // unsqueeze_4对应的是163,164,17是position_ids,162是pd_op.full_int_array
    unsqueeze_4({&pat.Tensor("position_ids"), &full_8()},
                {&pat.Tensor("unsqueeze_s_out_sin"), &pat.Tensor("xshape")});

    // gather_nd对应165,153是pd_op.squeeze,163是pd_op.unsqueeze
    pat.Tensor("gather_nd_out_sin") = gather_nd_1(
        pat.Tensor("squeeze_out_sin"), pat.Tensor("unsqueeze_s_out_sin"));

    // unsqueeze_2对应的是167,168,165是pd.gather)nd和pd_op.full_int_array
    unsqueeze_2({&pat.Tensor("gather_nd_out_sin"), &full_9()},
                {&pat.Tensor("unsqueeze_out_sin"), &pat.Tensor("xshape")});

    // multiply1对应的是169,第一个参数是q,第二个参数是pd_op.unsqueeze
    pat.Tensor("tmp_25") =
        multiply1(pat.Tensor("q"), pat.Tensor("unsqueeze_out_cos"));

    // slice_q对应的是172,129是q,170和171对应的是pd_op.full_int_array
    pat.Tensor("q_slice_out1") = slice_q(pat.Tensor("q"), full_1(), full_2());

    // slice_q_1对应的是175,129是q,173和174是pd_op.full_int_array
    pat.Tensor("q_slice_out2") = slice_q_1(pat.Tensor("q"), full_3(), full_4());

    // scale_op对应的是177,175对应的是slice,176对应的是pd_op.full
    scale_op({&pat.Tensor("q_slice_out2"), &full_op()},
             {{&pat.Tensor("scale_out")}});

    std::vector<const paddle::drr::Tensor *> combine_in;
    combine_in.push_back(&pat.Tensor("scale_out"));
    combine_in.push_back(&pat.Tensor("q_slice_out1"));
    // combine对应的是178,177是pd_op.scale,172是pd_op.slice
    combine(combine_in, {&pat.Tensor("combine_out")});

    // concat_op对应的是180,178是builtion.combine,179对应的是pd_op.full
    concat_op({&pat.Tensor("combine_out"), &full_op_3()},
              {&pat.Tensor("concat_out")});

    // multiply对应的是181,180对应的是concat,167对应的是pd_op.unsqueeze
    pat.Tensor("tmp_27") =
        multiply3(pat.Tensor("concat_out"), pat.Tensor("unsqueeze_out_sin"));

    // add对应182,169对应的multiply,181对应的是multiply
    pat.Tensor("out_q") = add(pat.Tensor("tmp_25"), pat.Tensor("tmp_27"));

    // multiply2对应的是183,132是k,160是pd.op.unsqueeze
    pat.Tensor("tmp_29") =
        multiply2(pat.Tensor("k"), pat.Tensor("unsqueeze_out_cos"));

    // slice_k对应的是186,132对应的是k,184和185对应的是pd.op.full_int_array
    pat.Tensor("k_slice_out1") = slice_k(pat.Tensor("k"), full_5(), full_6());

    // slice_k_1对应的是189,132对应的是k,然后两个pd_op.full_int_array
    pat.Tensor("k_slice_out2") =
        slice_k_1(pat.Tensor("k"), full_7(), full_14());

    // 191是scale_op_k,189对应的是slice,190对应的是pd_op.full
    scale_op_k({&pat.Tensor("k_slice_out2"), &full_op_1()},
               {{&pat.Tensor("scale_out_k")}});

    // 192是combine_k,191对应的是pd_scale.combine,186是pd_op.slice
    std::vector<const paddle::drr::Tensor *> combine_in_k;
    combine_in_k.push_back(&pat.Tensor("scale_out_k"));
    combine_in_k.push_back(&pat.Tensor("k_slice_out1"));
    combine_k(combine_in_k, {&pat.Tensor("combine_out_k")});

    // concat_op_k对应的是194,192为conbine,193为pd_op.full
    concat_op_k({&pat.Tensor("combine_out_k"), &full_op_2()},
                {&pat.Tensor("concat_out_k")});

    // tmp_31是195,concat_out_k是194,unsqueeze_out_sin是167
    pat.Tensor("tmp_31") =
        multiply4(pat.Tensor("concat_out_k"), pat.Tensor("unsqueeze_out_sin"));
    // tmp_29对应的%183,tmp_31对应的是%195
    pat.Tensor("out_k") = add_1(pat.Tensor("tmp_29"), pat.Tensor("tmp_31"));

    pat.RequireNativeCall([&](const paddle::drr::MatchContext &match_ctx) {
      auto check_axes = [&](const std::vector<int64_t> &axes) {
        std::vector<int64_t> expected_axes = {0, 2};
        if (axes.size() != expected_axes.size()) {
          return false;
        }
        for (size_t i = 0; i < axes.size(); ++i) {
          if (axes[i] != expected_axes[i]) {
            return false;
          }
        }
        return true;
      };
      auto axis = match_ctx.Attr<std::vector<int64_t>>("full_13_value");
      auto axis_2 = match_ctx.Attr<std::vector<int64_t>>("full_12_value");
      return check_axes(axis) && check_axes(axis_2);

      auto check_unsqueeze_axes = [&](const std::vector<int64_t> &axes) {
        std::vector<int64_t> expected_axes = {0};
        if (axes.size() != expected_axes.size()) {
          return false;
        }
        for (size_t i = 0; i < axes.size(); ++i) {
          if (axes[i] != expected_axes[i]) {
            return false;
          }
        }
        return true;
      };
      auto unsqueeze_axis =
          match_ctx.Attr<std::vector<int64_t>>("full_11_value");
      auto unsqueeze_axis_1 =
          match_ctx.Attr<std::vector<int64_t>>("full_10_value");
      auto unsqueeze_axis_2 =
          match_ctx.Attr<std::vector<int64_t>>("full_8_value");
      auto unsqueeze_axis_3 =
          match_ctx.Attr<std::vector<int64_t>>("full_9_value");

      return check_unsqueeze_axes(unsqueeze_axis) &&
             check_unsqueeze_axes(unsqueeze_axis_1) &&
             check_unsqueeze_axes(unsqueeze_axis_2) &&
             check_unsqueeze_axes(unsqueeze_axis_3);

      auto check_concat_axes = [&](const std::vector<int64_t> &axes) {
        std::vector<int64_t> expected_axes = {-1};
        if (axes.size() != expected_axes.size()) {
          return false;
        }
        for (size_t i = 0; i < axes.size(); ++i) {
          if (axes[i] != expected_axes[i]) {
            return false;
          }
        }
        return true;
      };
      auto concat_axis = match_ctx.Attr<std::vector<int64_t>>("full_op_3");
      auto concat_axis_1 = match_ctx.Attr<std::vector<int64_t>>("full_op_2");
      return check_concat_axes(concat_axis) && check_concat_axes(concat_axis_1);
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &fused_rotary_position_embedding =
        res.Op(paddle::dialect::FusedRotaryPositionEmbeddingOp::name(),
               {
                   {"use_neox_rotary_style", res.BoolAttr(true)},
                   {"time_major", res.BoolAttr(false)},
                   {"rotary_emb_base", res.Float32Attr(10000.0)},
               });

    fused_rotary_position_embedding(
        {&res.Tensor("q"),
         &res.Tensor("k"),
         &res.InputNoneTensor(),
         &res.Tensor("sin"),
         &res.Tensor("cos"),
         &res.Tensor("position_ids")},
        {&res.Tensor("out_q"), &res.Tensor("out_k"), &res.OutputNoneTensor()});
  }
};
class FusedRotaryPositionEmbeddingPass : public pir::PatternRewritePass {
 public:
  FusedRotaryPositionEmbeddingPass()
      : pir::PatternRewritePass("fused_rotary_position_embedding_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<FusedRotaryPositionEmbeddingPattern>(context));
    return ps;
  }
};
}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateFusedRotaryPositionEmbeddingPass() {
  return std::make_unique<FusedRotaryPositionEmbeddingPass>();
}
}  // namespace pir
REGISTER_IR_PASS(fused_rotary_position_embedding_pass,
                 FusedRotaryPositionEmbeddingPass);
