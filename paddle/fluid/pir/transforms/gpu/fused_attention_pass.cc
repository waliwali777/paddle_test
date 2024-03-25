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

#include "paddle/fluid/pir/transforms/gpu/multihead_matmul_fuse_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class FlashAttnPatternQscaleCast : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override {
      return "FlashAttnPatternQscaleAlibiCast";
  }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern src = ctx->SourcePattern();
    // check the transpose, 
    // first pattern with alibi + cast
    // q[b, s, head, head_dim] -> transpose -> q[b, head, s, head_dim] -> scale
    const auto &transpose_q = src.Op("pd_op.transpose");
    src.Tensor("q_transpose_out") = transpose_q(src.Tensor("q"));
    const auto &scale_q = src.Op("pd_op.scale");
    const auto &full_scale = src.Op("pd_op.full", {{"value", src.Attr("scale_q_value")}});
    src.Tensor("q_scale_out") = scale_q(src.Tensor("q_transpose_out"), full_scale());
    // k[b, s, head, head_dim] -> transpose -> k[b, head, s, head_dim]
    // k[b, head, s, head_dim] -> transpose -> k[b, head, head_dim, s]
    const auto &transpose_k = src.Op("pd_op.transpose");
    src.Tensor("k_transpose_out") = transpose_k(src.Tensor("k"));
    const auto &transpose_k2 = src.Op("pd_op.transpose");
    src.Tensor("k_transpose2_out") = transpose_k2(src.Tensor("k_transpose_out"));
    // v[b, s, head, head_dim] -> transpose -> v[b, head, s, head_dim]
    const auto &transpose_v = src.Op("pd_op.transpose");
    src.Tensor("v_transpose_out") = transpose_v(src.Tensor("v"));
    // qk
    const auto &qk_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_qk_transpose_x")},
                {"transpose_y", src.Attr("matmul_qk_transpose_y")}});
    src.Tensor("qk_out") = 
        qk_matmul(src.Tensor("q_scale_out"), src.Tensor("k_transpose2_out"));

    // mask
    const auto &mask_reshape = src.Op("pd_op.reshape");
    const auto &mask_full = src.Op("pd_op.full_int_array");
    mask_reshape({&src.Tensor("mask"), &mask_full()},
                 {&src.Tensor("mask_reshape"), &src.Tensor("mask_shape")});
    const auto &mask_add = src.Op("pd_op.add");
    src.Tensor("mask_add_out") =
        mask_add(src.Tensor("qk_out"), src.Tensor("mask_reshape"));

    // cast + softmax + cast
    const auto &softmax_cast1 = src.Op("pd_op.cast");
    src.Tensor("softmax_cast1_out") = softmax_cast1(src.Tensor("mask_add_out"));
    const auto &softmax = src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_axis")}});
    src.Tensor("softmax_out") = softmax(src.Tensor("softmax_cast1_out"));
    const auto &softmax_cast2 = src.Op("pd_op.cast");
    src.Tensor("softmax_cast2_out") = softmax_cast2(src.Tensor("softmax_out"));

    // o
    const auto &context_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("context_matmul_transpose_x")},
                {"transpose_y", src.Attr("context_matmul_transpose_y")}});
    src.Tensor("context_matmul_out") = context_matmul(
        src.Tensor("softmax_cast2_out"), src.Tensor("v_transpose_out"));
    const auto &o_transpose = src.Op("pd_op.transpose");
    src.Tensor("out") = o_transpose(src.Tensor("context_matmul_out"));

    //Constraints
    src.RequireNativeCall([](const paddle::drr::MatchContext &match_ctx)
                              -> bool {
      // softmax                            
      const auto &softmax_axis = match_ctx.Attr<int>("softmax_axis");
      if (softmax_axis != -1 && softmax_axis != 3) return false;
      //matmul transpose
      bool matmul_qk_transpose_x = match_ctx.Attr<bool>("matmul_1_transpose_x");
      bool matmul_qk_transpose_y = match_ctx.Attr<bool>("matmul_1_transpose_y");
      if (matmul_qk_transpose_x || matmul_qk_transpose_y) return false;

      bool matmul_o_transpose_x = match_ctx.Attr<bool>("context_matmul_transpose_x");
      bool matmul_o_transpose_y = match_ctx.Attr<bool>("context_matmul_transpose_y");
      if (matmul_o_transpose_x || matmul_o_transpose_y) return false;
      // tensor shape
      auto q_transpose_out =
          pir::GetShapeFromValue(match_ctx.Tensor("q_transpose_out"));
      auto k_transpose_out =
          pir::GetShapeFromValue(match_ctx.Tensor("k_transpose_out"));
      auto v_transpose_out =
          pir::GetShapeFromValue(match_ctx.Tensor("v_transpose_out"));
      if (q_transpose_out.size() != 4 || k_transpose_out.size() != 4 || v_transpose_out.size() != 4 ||
          !(q_transpose_out.at(0) == k.transpose_out.at(0) && k.transpose_out.at(0) == v.transpose.at(0)) || 
          !(q_transpose_out.at(1) == k.transpose_out.at(1) && k.transpose_out.at(1) == v.transpose.at(1)) ||
          !(q_transpose_out.at(3) == k.transpose_out.at(3) && k.transpose_out.at(3) == v.transpose.at(3))) {
            return false;
          }
      // add shape
      auto alibi_add = pir::GetShapeFromValue(match_ctx.Tensor("alibi_reshape"));
      auto mask_add = pir::GetShapeFromValue(match_ctx.Tensor("mask_reshape"));
      if (alibi_add.size() != 4 || mask_add.size() != 4) {
          return false;
      }


      return true;
    });

    //
    // Result Pattern.
    //
    paddle::drr::ResultPattern res = src.ResultPattern();
    // flash_attn impl
    const auto &flash_attn = res.Op("pd_op.FlashAttn", {{}});
    flash_attn({&res.Tensor("q"),
                &res.Tensor("k"),
                &res.Tensor("v"),
                })
    // prepare for flash_attn
    // reference flash_attn
    // bool flash_attn_fwd(const void * const q,         // batch_size x seqlen_q x num_heads x head_size
    //                const void * const k,         // batch_size x seqlen_k x num_heads_k x head_size
    //                const void * const v,         // batch_size x seqlen_k x num_heads_k x head_size
    //                void * const rng_state,
    //                void * const out,
    //                void * const softmax_ptr,
    //                void * const softmax_lse_ptr,
                    // const int batch_size,
                    // const int seqlen_q,
                    // const int seqlen_k,
                    // const int seqlen_q_rounded,
                    // const int seqlen_k_rounded,
                    // const int num_heads,
                    // const int num_heads_k,
                    // const int head_size,
                    // const int head_size_rounded,
                    // const float p_dropout,
                    // const float softmax_scale,
                    // const float softmax_unscale,
                    // const bool is_causal,
                    // const bool return_softmax,
                    // const bool is_bf16,
                    // cudaStream_t stream,
                    // uint64_t seed,
                    // uint64_t offset,
                    // const void * const attn_mask,
                    // const int64_t * const mask_dims,
                    // const void * const attn_mask_start_row_indices,
                    // const int64_t * const attn_mask_start_row_indices_dims,
                    // const int attn_mask_start_row);
  }
};

class FlashAttnPatternQscaleNoCast : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override {
      return "FlashAttnPatternQscaleAlibiNoCast";
  }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern src = ctx->SourcePattern();
    // check the transpose, 
    // first pattern with alibi + cast
    // q[b, s, head, head_dim] -> transpose -> q[b, head, s, head_dim] -> scale
    const auto &transpose_q = src.Op("pd_op.transpose");
    src.Tensor("q_transpose_out") = transpose_q(src.Tensor("q"));
    const auto &scale_q = src.Op("pd_op.scale");
    const auto &full_scale = src.Op("pd_op.full", {{"value", src.Attr("scale_q_value")}});
    src.Tensor("q_scale_out") = scale_q(src.Tensor("q_transpose_out"), full_scale());
    // k[b, s, head, head_dim] -> transpose -> k[b, head, s, head_dim]
    // k[b, head, s, head_dim] -> transpose -> k[b, head, head_dim, s]
    const auto &transpose_k = src.Op("pd_op.transpose");
    src.Tensor("k_transpose_out") = transpose_k(src.Tensor("k"));
    const auto &transpose_k2 = src.Op("pd_op.transpose");
    src.Tensor("k_transpose2_out") = transpose_k2(src.Tensor("k_transpose_out"));
    // v[b, s, head, head_dim] -> transpose -> v[b, head, s, head_dim]
    const auto &transpose_v = src.Op("pd_op.transpose");
    src.Tensor("v_transpose_out") = transpose_v(src.Tensor("v"));
    // qk
    const auto &qk_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_qk_transpose_x")},
                {"transpose_y", src.Attr("matmul_qk_transpose_y")}});
    src.Tensor("qk_out") = 
        qk_matmul(src.Tensor("q_scale_out"), src.Tensor("k_transpose2_out"));

    // mask
    const auto &mask_reshape = src.Op("pd_op.reshape");
    const auto &mask_full = src.Op("pd_op.full_int_array");
    mask_reshape({&src.Tensor("mask"), &mask_full()},
                 {&src.Tensor("mask_reshape"), &src.Tensor("mask_shape")});
    const auto &mask_add = src.Op("pd_op.add");
    src.Tensor("mask_add_out") =
        mask_add(src.Tensor("qk_out"), src.Tensor("mask_reshape"));

    // softmax
    const auto &softmax = src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_axis")}});
    src.Tensor("softmax_out") = softmax(src.Tensor("mask_add_out"));

    // o
    const auto &context_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("context_matmul_transpose_x")},
                {"transpose_y", src.Attr("context_matmul_transpose_y")}});
    src.Tensor("context_matmul_out") = context_matmul(
        src.Tensor("softmax_cast2_out"), src.Tensor("v_transpose_out"));
    const auto &o_transpose = src.Op("pd_op.transpose");
    src.Tensor("out") = o_transpose(src.Tensor("context_matmul_out"));

    //Constraints
    src.RequireNativeCall([](const paddle::drr::MatchContext &match_ctx)
                              -> bool {
      // softmax                            
      const auto &softmax_axis = match_ctx.Attr<int>("softmax_axis");
      if (softmax_axis != -1 && softmax_axis != 3) return false;
      //matmul transpose
      bool matmul_qk_transpose_x = match_ctx.Attr<bool>("matmul_1_transpose_x");
      bool matmul_qk_transpose_y = match_ctx.Attr<bool>("matmul_1_transpose_y");
      if (matmul_qk_transpose_x || matmul_qk_transpose_y) return false;

      bool matmul_o_transpose_x = match_ctx.Attr<bool>("context_matmul_transpose_x");
      bool matmul_o_transpose_y = match_ctx.Attr<bool>("context_matmul_transpose_y");
      if (matmul_o_transpose_x || matmul_o_transpose_y) return false;
      // tensor shape
      auto q_transpose_out =
          pir::GetShapeFromValue(match_ctx.Tensor("q_transpose_out"));
      auto k_transpose_out =
          pir::GetShapeFromValue(match_ctx.Tensor("k_transpose_out"));
      auto v_transpose_out =
          pir::GetShapeFromValue(match_ctx.Tensor("v_transpose_out"));
      if (q_transpose_out.size() != 4 || k_transpose_out.size() != 4 || v_transpose_out.size() != 4 ||
          !(q_transpose_out.at(0) == k.transpose_out.at(0) && k.transpose_out.at(0) == v.transpose.at(0)) || 
          !(q_transpose_out.at(1) == k.transpose_out.at(1) && k.transpose_out.at(1) == v.transpose.at(1)) ||
          !(q_transpose_out.at(3) == k.transpose_out.at(3) && k.transpose_out.at(3) == v.transpose.at(3))) {
            return false;
          }
      // add shape
      auto alibi_add = pir::GetShapeFromValue(match_ctx.Tensor("alibi_reshape"));
      auto mask_add = pir::GetShapeFromValue(match_ctx.Tensor("mask_reshape"));
      if (alibi_add.size() != 4 || mask_add.size() != 4) {
          return false;
      }


      return true;
    });

    //
    // Result Pattern.
    //
    paddle::drr::ResultPattern res = src.ResultPattern();
    // flash_attn impl
    const auto &flash_attn = res.Op("pd_op.FlashAttn", {{}});
    flash_attn({&res.Tensor("q"),
                &res.Tensor("k"),
                &res.Tensor("v"),
                })
  }
};

class FlashAttnPatternOutscaleCast : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override {
      return "FlashAttnPatternOutscaleAlibiCast";
  }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern src = ctx->SourcePattern();
    // check the transpose, 
    // first pattern with alibi + cast
    // q[b, s, head, head_dim] -> transpose -> q[b, head, s, head_dim] -> scale
    const auto &transpose_q = src.Op("pd_op.transpose");
    src.Tensor("q_transpose_out") = transpose_q(src.Tensor("q"));
    // k[b, s, head, head_dim] -> transpose -> k[b, head, s, head_dim]
    // k[b, head, s, head_dim] -> transpose -> k[b, head, head_dim, s]
    const auto &transpose_k = src.Op("pd_op.transpose");
    src.Tensor("k_transpose_out") = transpose_k(src.Tensor("k"));
    const auto &transpose_k2 = src.Op("pd_op.transpose");
    src.Tensor("k_transpose2_out") = transpose_k2(src.Tensor("k_transpose_out"));
    // v[b, s, head, head_dim] -> transpose -> v[b, head, s, head_dim]
    const auto &transpose_v = src.Op("pd_op.transpose");
    src.Tensor("v_transpose_out") = transpose_v(src.Tensor("v"));
    // qk
    const auto &qk_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_qk_transpose_x")},
                {"transpose_y", src.Attr("matmul_qk_transpose_y")}});
    src.Tensor("qk_out") = 
        qk_matmul(src.Tensor("q_transpose_out"), src.Tensor("k_transpose2_out"));
    const auto &scale_out = src.Op("pd_op.scale");
    const auto &full_scale = src.Op("pd_op.full", {{"value", src.Attr("scale_out_value")}});
    src.Tensor("qk_scale_out") = scale_out(src.Tensor("qk_out"), full_scale());

    // mask
    const auto &mask_reshape = src.Op("pd_op");
    const auto &mask_full = src.Op("pd_op.full_int_array");
    mask_reshape({&src.Tensor("mask"), &mask_full()},
                 {&src.Tensor("mask_reshape"), &src.Tensor("mask_shape")});
    const auto &mask_add = src.Op("pd_op.add");
    src.Tensor("mask_add_out") =
        mask_add(src.Tensor("qk_scale_out"), src.Tensor("mask_reshape"));

    // cast + softmax + cast
    const auto &softmax_cast1 = src.Op("pd_op.cast");
    src.Tensor("softmax_cast1_out") = softmax_cast1(src.Tensor("mask_add_out"));
    const auto &softmax = src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_axis")}});
    src.Tensor("softmax_out") = softmax(src.Tensor("softmax_cast1_out"));
    const auto &softmax_cast2 = src.Op("pd_op.cast");
    src.Tensor("softmax_cast2_out") = softmax_cast2(src.Tensor("softmax_out"));

    // o
    const auto &context_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("context_matmul_transpose_x")},
                {"transpose_y", src.Attr("context_matmul_transpose_y")}});
    src.Tensor("context_matmul_out") = context_matmul(
        src.Tensor("softmax_cast2_out"), src.Tensor("v_transpose_out"));
    const auto &o_transpose = src.Op("pd_op.transpose");
    src.Tensor("out") = o_transpose(src.Tensor("context_matmul_out"));

    //Constraints
    src.RequireNativeCall([](const paddle::drr::MatchContext &match_ctx)
                              -> bool {
      // softmax                            
      const auto &softmax_axis = match_ctx.Attr<int>("softmax_axis");
      if (softmax_axis != -1 && softmax_axis != 3) return false;
      //matmul transpose
      bool matmul_qk_transpose_x = match_ctx.Attr<bool>("matmul_1_transpose_x");
      bool matmul_qk_transpose_y = match_ctx.Attr<bool>("matmul_1_transpose_y");
      if (matmul_qk_transpose_x || matmul_qk_transpose_y) return false;

      bool matmul_o_transpose_x = match_ctx.Attr<bool>("context_matmul_transpose_x");
      bool matmul_o_transpose_y = match_ctx.Attr<bool>("context_matmul_transpose_y");
      if (matmul_o_transpose_x || matmul_o_transpose_y) return false;
      // tensor shape
      auto q_transpose_out =
          pir::GetShapeFromValue(match_ctx.Tensor("q_transpose_out"));
      auto k_transpose_out =
          pir::GetShapeFromValue(match_ctx.Tensor("k_transpose_out"));
      auto v_transpose_out =
          pir::GetShapeFromValue(match_ctx.Tensor("v_transpose_out"));
      if (q_transpose_out.size() != 4 || k_transpose_out.size() != 4 || v_transpose_out.size() != 4 ||
          !(q_transpose_out.at(0) == k.transpose_out.at(0) && k.transpose_out.at(0) == v.transpose.at(0)) || 
          !(q_transpose_out.at(1) == k.transpose_out.at(1) && k.transpose_out.at(1) == v.transpose.at(1)) ||
          !(q_transpose_out.at(3) == k.transpose_out.at(3) && k.transpose_out.at(3) == v.transpose.at(3))) {
            return false;
          }
      // add shape
      auto alibi_add = pir::GetShapeFromValue(match_ctx.Tensor("alibi_reshape"));
      auto mask_add = pir::GetShapeFromValue(match_ctx.Tensor("mask_reshape"));
      if (alibi_add.size() != 4 || mask_add.size() != 4) {
          return false;
      }


      return true;
    });

    //
    // Result Pattern.
    //
    paddle::drr::ResultPattern res = src.ResultPattern();
    // flash_attn impl
    const auto &flash_attn = res.Op("pd_op.FlashAttn", {{}});
    flash_attn({&res.Tensor("q"),
                &res.Tensor("k"),
                &res.Tensor("v"),
                })
    // prepare for flash_attn
    // reference flash_attn
    // bool flash_attn_fwd(const void * const q,         // batch_size x seqlen_q x num_heads x head_size
    //                const void * const k,         // batch_size x seqlen_k x num_heads_k x head_size
    //                const void * const v,         // batch_size x seqlen_k x num_heads_k x head_size
    //                void * const rng_state,
    //                void * const out,
    //                void * const softmax_ptr,
    //                void * const softmax_lse_ptr,
                    // const int batch_size,
                    // const int seqlen_q,
                    // const int seqlen_k,
                    // const int seqlen_q_rounded,
                    // const int seqlen_k_rounded,
                    // const int num_heads,
                    // const int num_heads_k,
                    // const int head_size,
                    // const int head_size_rounded,
                    // const float p_dropout,
                    // const float softmax_scale,
                    // const float softmax_unscale,
                    // const bool is_causal,
                    // const bool return_softmax,
                    // const bool is_bf16,
                    // cudaStream_t stream,
                    // uint64_t seed,
                    // uint64_t offset,
                    // const void * const attn_mask,
                    // const int64_t * const mask_dims,
                    // const void * const attn_mask_start_row_indices,
                    // const int64_t * const attn_mask_start_row_indices_dims,
                    // const int attn_mask_start_row);
  }
};

class FlashAttnPatternOutscaleNoCast : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override {
      return "FlashAttnPatternOutscaleAlibiNoCast";
  }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern src = ctx->SourcePattern();
    // check the transpose, 
    // first pattern with alibi + cast
    // q[b, s, head, head_dim] -> transpose -> q[b, head, s, head_dim] -> scale
    const auto &transpose_q = src.Op("pd_op.transpose");
    src.Tensor("q_transpose_out") = transpose_q(src.Tensor("q"));
    // k[b, s, head, head_dim] -> transpose -> k[b, head, s, head_dim]
    // k[b, head, s, head_dim] -> transpose -> k[b, head, head_dim, s]
    const auto &transpose_k = src.Op("pd_op.transpose");
    src.Tensor("k_transpose_out") = transpose_k(src.Tensor("k"));
    const auto &transpose_k2 = src.Op("pd_op.transpose");
    src.Tensor("k_transpose2_out") = transpose_k2(src.Tensor("k_transpose_out"));
    // v[b, s, head, head_dim] -> transpose -> v[b, head, s, head_dim]
    const auto &transpose_v = src.Op("pd_op.transpose");
    src.Tensor("v_transpose_out") = transpose_v(src.Tensor("v"));
    // qk
    const auto &qk_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("matmul_qk_transpose_x")},
                {"transpose_y", src.Attr("matmul_qk_transpose_y")}});
    src.Tensor("qk_out") = 
        qk_matmul(src.Tensor("q_transpose_out"), src.Tensor("k_transpose2_out"));
    const auto &scale_out = src.Op("pd_op.scale");
    const auto &full_scale = src.Op("pd_op.full", {{"value", src.Attr("scale_out_value")}});
    src.Tensor("qk_scale_out") = scale_out(src.Tensor("qk_out"), full_scale());

    // mask
    const auto &mask_reshape = src.Op("pd_op");
    const auto &mask_full = src.Op("pd_op.full_int_array");
    mask_reshape({&src.Tensor("mask"), &mask_full()},
                 {&src.Tensor("mask_reshape"), &src.Tensor("mask_shape")});
    const auto &mask_add = src.Op("pd_op.add");
    src.Tensor("mask_add_out") =
        mask_add(src.Tensor("qk_scale_out"), src.Tensor("mask_reshape"));

    // softmax
    const auto &softmax = src.Op("pd_op.softmax", {{"axis", src.Attr("softmax_axis")}});
    src.Tensor("softmax_out") = softmax(src.Tensor("mask_add_out"));

    // o
    const auto &context_matmul =
        src.Op("pd_op.matmul",
               {{"transpose_x", src.Attr("context_matmul_transpose_x")},
                {"transpose_y", src.Attr("context_matmul_transpose_y")}});
    src.Tensor("context_matmul_out") = context_matmul(
        src.Tensor("softmax_cast2_out"), src.Tensor("v_transpose_out"));
    const auto &o_transpose = src.Op("pd_op.transpose");
    src.Tensor("out") = o_transpose(src.Tensor("context_matmul_out"));

    //Constraints
    src.RequireNativeCall([](const paddle::drr::MatchContext &match_ctx)
                              -> bool {
      // softmax                            
      const auto &softmax_axis = match_ctx.Attr<int>("softmax_axis");
      if (softmax_axis != -1 && softmax_axis != 3) return false;
      //matmul transpose
      bool matmul_qk_transpose_x = match_ctx.Attr<bool>("matmul_1_transpose_x");
      bool matmul_qk_transpose_y = match_ctx.Attr<bool>("matmul_1_transpose_y");
      if (matmul_qk_transpose_x || matmul_qk_transpose_y) return false;

      bool matmul_o_transpose_x = match_ctx.Attr<bool>("context_matmul_transpose_x");
      bool matmul_o_transpose_y = match_ctx.Attr<bool>("context_matmul_transpose_y");
      if (matmul_o_transpose_x || matmul_o_transpose_y) return false;
      // tensor shape
      auto q_transpose_out =
          pir::GetShapeFromValue(match_ctx.Tensor("q_transpose_out"));
      auto k_transpose_out =
          pir::GetShapeFromValue(match_ctx.Tensor("k_transpose_out"));
      auto v_transpose_out =
          pir::GetShapeFromValue(match_ctx.Tensor("v_transpose_out"));
      if (q_transpose_out.size() != 4 || k_transpose_out.size() != 4 || v_transpose_out.size() != 4 ||
          !(q_transpose_out.at(0) == k.transpose_out.at(0) && k.transpose_out.at(0) == v.transpose.at(0)) || 
          !(q_transpose_out.at(1) == k.transpose_out.at(1) && k.transpose_out.at(1) == v.transpose.at(1)) ||
          !(q_transpose_out.at(3) == k.transpose_out.at(3) && k.transpose_out.at(3) == v.transpose.at(3))) {
            return false;
          }
      // add shape
      auto alibi_add = pir::GetShapeFromValue(match_ctx.Tensor("alibi_reshape"));
      auto mask_add = pir::GetShapeFromValue(match_ctx.Tensor("mask_reshape"));
      if (alibi_add.size() != 4 || mask_add.size() != 4) {
          return false;
      }


      return true;
    });

    //
    // Result Pattern.
    //
    paddle::drr::ResultPattern res = src.ResultPattern();
    // flash_attn impl
    const auto &flash_attn = res.Op("pd_op.FlashAttn", {{}});
    flash_attn({&res.Tensor("q"),
                &res.Tensor("k"),
                &res.Tensor("v"),
                })
  }
};

class FlashAttnFusePass : public pir::PatternRewritePass {
 public:
  FlashAttnFusePass()
      : pir::PatternRewritePass("flash_attn_fuse_pass", 2) {}
  
  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<FlashAttnPatternQscaleCast>(context));
    ps.Add(paddle::drr::Create<FlashAttnPatternQscaleNoCast>(context));
    ps.Add(paddle::drr::Create<FlashAttnPatternOutscaleCast>(context));
    ps.Add(paddle::drr::Create<FlashAttnPatternOutscaleNoCast>(context));

    return ps;
  }
};

}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateFlashAttnFusePass() {
  return std::make_unique<FlashAttnFusePass>();
}
}  // namespace pir

REGISTER_IR_PASS(flash_attn_fuse_pass, FlashAttnFusePass);
