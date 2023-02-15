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

#include "paddle/fluid/framework/ir/fused_feedforward_pass.h"

#include <string>
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void FusedFeedForwardPass::ApplyImpl(ir::Graph *graph) const {
  FusePassBase::Init(scope_name, graph);
  // FusedFeedForwardFwd(ir::Graph *graph, bool pre_layer_norm, bool
  // add_residual) pre_layer_norm and add_residual can't both be false!
  graph = FusedFeedForwardFwd(graph, true, true);
  graph = FusedFeedForwardFwd(graph, true, false);
  graph = FusedFeedForwardFwd(graph, false, true);

  graph = FusedFeedForwardBwd(graph, true, true);
  graph = FusedFeedForwardBwd(graph, true, false);
  graph = FusedFeedForwardBwd(graph, false, true);
}

ir::Graph *FusedFeedForwardPass::FusedFeedForwardFwd(ir::Graph *graph,
                                                     bool pre_layer_norm,
                                                     bool add_residual) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  GraphPatternDetector gpd;
  auto *x = gpd.mutable_pattern()
                ->NewNode(patterns::PDNodeName(scope_name, "x"))
                ->AsInput();
  if (pre_layer_norm) {
    x->assert_is_op_input("layer_norm", "X");
  } else {
    x->assert_is_op_input("matmul_v2", "X");
  }

  // 1. layer_norm -> linear1 -> activation -> dropout1 -> linear2 -> dropout2
  // -> residual_add
  // 2. layer_norm -> linear1 -> activation -> dropout1 -> linear2 -> dropout2
  // 3. linear1 -> activation -> dropout1 -> linear2 -> dropout2 -> residual_add
  // -> layer_norm
  // 4. linear1 -> activation -> dropout1 -> linear2 -> dropout2 -> layer_norm
  patterns::FusedFeedForwardFwd fused_feedforward_pattern(gpd.mutable_pattern(),
                                                          scope_name);
  std::unordered_set<std::string> act_types = {"gelu", "relu"};
  fused_feedforward_pattern(x, act_types, pre_layer_norm, add_residual);

  int found_fused_feedforward_fwd_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle feed_forward forward fusion";

    // LayerNorm
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_op, layer_norm_op, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_bias, layer_norm_bias, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_scale, layer_norm_scale, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_out, layer_norm_out, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_mean, layer_norm_mean, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_variance, layer_norm_variance, fused_feedforward_pattern);
    // Linear1
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_op_1, matmul_op_1, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_w_1, matmul_w_1, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_out_1, matmul_out_1, fused_feedforward_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_op_1, ele_add_op_1, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_bias_1, ele_add_bias_1, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_out_1, ele_add_out_1, fused_feedforward_pattern);
    // Dropout1
    GET_IR_NODE_FROM_SUBGRAPH(
        dropout_op_1, dropout_op_1, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        dropout_mask_1, dropout_mask_1, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        dropout_out_1, dropout_out_1, fused_feedforward_pattern);
    // Activation
    GET_IR_NODE_FROM_SUBGRAPH(act_op, act_op, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act_out, act_out, fused_feedforward_pattern);
    // Linear2
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_op_2, matmul_op_2, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_w_2, matmul_w_2, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_out_2, matmul_out_2, fused_feedforward_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_op_2, ele_add_op_2, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_bias_2, ele_add_bias_2, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_out_2, ele_add_out_2, fused_feedforward_pattern);
    // Dropout2
    GET_IR_NODE_FROM_SUBGRAPH(
        dropout_op_2, dropout_op_2, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        dropout_mask_2, dropout_mask_2, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        dropout_out_2, dropout_out_2, fused_feedforward_pattern);

    if (PADDLE_GET_CONST(bool, dropout_op_1->Op()->GetAttr("is_test")) !=
        PADDLE_GET_CONST(bool, dropout_op_2->Op()->GetAttr("is_test"))) {
      LOG(WARNING)
          << "Dropout 1 and dropout 2 attribute is_test set different values. "
          << "Skip fused_feedforward pattern replacement.";
      return;
    }

    OpDesc fused_feedforward_op_desc(layer_norm_op->Op()->Block());

    fused_feedforward_op_desc.SetType("fused_feedforward");
    fused_feedforward_op_desc.SetInput("X", {subgraph.at(x)->Name()});
    fused_feedforward_op_desc.SetInput("Linear1Weight", {matmul_w_1->Name()});
    fused_feedforward_op_desc.SetInput("Linear1Bias", {ele_add_bias_1->Name()});
    fused_feedforward_op_desc.SetInput("Linear2Weight", {matmul_w_2->Name()});
    fused_feedforward_op_desc.SetInput("Linear2Bias", {ele_add_bias_2->Name()});
    if (pre_layer_norm) {
      fused_feedforward_op_desc.SetInput("Ln1Scale",
                                         {layer_norm_scale->Name()});
      fused_feedforward_op_desc.SetInput("Ln1Bias", {layer_norm_bias->Name()});
      fused_feedforward_op_desc.SetOutput("Ln1Mean", {layer_norm_mean->Name()});
      fused_feedforward_op_desc.SetOutput("Ln1Variance",
                                          {layer_norm_variance->Name()});
      fused_feedforward_op_desc.SetOutput("Ln1Out", {layer_norm_out->Name()});
      fused_feedforward_op_desc.SetAttr(
          "ln1_epsilon", layer_norm_op->Op()->GetAttr("epsilon"));
      if (!add_residual) {
        fused_feedforward_op_desc.SetOutput("Out", {dropout_out_2->Name()});
      } else {
        GET_IR_NODE_FROM_SUBGRAPH(
            ele_add_out_3, ele_add_out_3, fused_feedforward_pattern);
        fused_feedforward_op_desc.SetOutput("Out", {ele_add_out_3->Name()});
      }
    } else {
      fused_feedforward_op_desc.SetInput("Ln2Scale",
                                         {layer_norm_scale->Name()});
      fused_feedforward_op_desc.SetInput("Ln2Bias", {layer_norm_bias->Name()});
      fused_feedforward_op_desc.SetOutput("Ln2Mean", {layer_norm_mean->Name()});
      fused_feedforward_op_desc.SetOutput("Ln2Variance",
                                          {layer_norm_variance->Name()});
      fused_feedforward_op_desc.SetAttr(
          "ln2_epsilon", layer_norm_op->Op()->GetAttr("epsilon"));
      fused_feedforward_op_desc.SetOutput("Out", {layer_norm_out->Name()});
    }
    fused_feedforward_op_desc.SetOutput("Dropout1Mask",
                                        {dropout_mask_1->Name()});
    fused_feedforward_op_desc.SetOutput("Dropout2Mask",
                                        {dropout_mask_2->Name()});
    fused_feedforward_op_desc.SetOutput("Linear1Out", {ele_add_out_1->Name()});
    fused_feedforward_op_desc.SetOutput("Dropout1Out", {dropout_out_1->Name()});
    fused_feedforward_op_desc.SetOutput("Dropout2Out", {dropout_out_2->Name()});

    fused_feedforward_op_desc.SetAttr("pre_layer_norm", pre_layer_norm);
    fused_feedforward_op_desc.SetAttr("act_method", act_op->Op()->Type());
    fused_feedforward_op_desc.SetAttr(
        "dropout1_rate", dropout_op_1->Op()->GetAttr("dropout_prob"));
    fused_feedforward_op_desc.SetAttr(
        "dropout2_rate", dropout_op_2->Op()->GetAttr("dropout_prob"));
    fused_feedforward_op_desc.SetAttr(
        "dropout1_implementation",
        dropout_op_1->Op()->GetAttr("dropout_implementation"));
    fused_feedforward_op_desc.SetAttr(
        "dropout2_implementation",
        dropout_op_2->Op()->GetAttr("dropout_implementation"));
    // These attributes set default value
    fused_feedforward_op_desc.SetAttr(
        "is_test",
        PADDLE_GET_CONST(bool, dropout_op_1->Op()->GetAttr("is_test")));
    fused_feedforward_op_desc.SetAttr("dropout1_fix_seed", false);
    fused_feedforward_op_desc.SetAttr("dropout2_fix_seed", false);
    fused_feedforward_op_desc.SetAttr("dropout1_seed", 0);
    fused_feedforward_op_desc.SetAttr("dropout2_seed", 0);
    fused_feedforward_op_desc.SetAttr("add_residual", add_residual);
    // fused_feedforward_op_desc.SetAttr("ring_id", {});

    auto fused_feedforward_node = g->CreateOpNode(&fused_feedforward_op_desc);

    IR_NODE_LINK_TO(subgraph.at(x), fused_feedforward_node);
    IR_NODE_LINK_TO(matmul_w_1, fused_feedforward_node);
    IR_NODE_LINK_TO(ele_add_bias_1, fused_feedforward_node);
    IR_NODE_LINK_TO(matmul_w_2, fused_feedforward_node);
    IR_NODE_LINK_TO(ele_add_bias_2, fused_feedforward_node);
    IR_NODE_LINK_TO(layer_norm_scale, fused_feedforward_node);
    IR_NODE_LINK_TO(layer_norm_bias, fused_feedforward_node);

    IR_NODE_LINK_TO(fused_feedforward_node, layer_norm_mean);
    IR_NODE_LINK_TO(fused_feedforward_node, layer_norm_variance);
    IR_NODE_LINK_TO(fused_feedforward_node, dropout_mask_1);
    IR_NODE_LINK_TO(fused_feedforward_node, dropout_mask_2);
    IR_NODE_LINK_TO(fused_feedforward_node, ele_add_out_1);
    IR_NODE_LINK_TO(fused_feedforward_node, dropout_out_1);
    IR_NODE_LINK_TO(fused_feedforward_node, dropout_out_2);
    if (!pre_layer_norm) {
      IR_NODE_LINK_TO(fused_feedforward_node, layer_norm_out);
    } else {
      if (add_residual) {
        // Residual Add, dispensable
        GET_IR_NODE_FROM_SUBGRAPH(
            ele_add_out_3, ele_add_out_3, fused_feedforward_pattern);
        IR_NODE_LINK_TO(fused_feedforward_node, ele_add_out_3);
      } else {
        IR_NODE_LINK_TO(fused_feedforward_node, dropout_out_2);
      }
    }

    std::unordered_set<const Node *> nodes_to_remove = {layer_norm_op,
                                                        matmul_op_1,
                                                        ele_add_op_1,
                                                        dropout_op_1,
                                                        act_op,
                                                        matmul_op_2,
                                                        ele_add_op_2,
                                                        dropout_op_2};
    if (add_residual) {
      // Residual
      GET_IR_NODE_FROM_SUBGRAPH(
          ele_add_op_3, ele_add_op_3, fused_feedforward_pattern);
      nodes_to_remove.insert(ele_add_op_3);
    }
    GraphSafeRemoveNodes(g, nodes_to_remove);
    found_fused_feedforward_fwd_count++;
  };

  gpd(graph, handler);
  AddStatis(found_fused_feedforward_fwd_count);
  return graph;
}

ir::Graph *FusedFeedForwardPass::FusedFeedForwardBwd(ir::Graph *graph,
                                                     bool pre_layer_norm,
                                                     bool add_residual) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  const std::string scope_name("fused_feed_forward_bwd_pattern");

  // 1. residual_add_grad -> dropout2_grad -> linear2_grad -> dropout1_grad ->
  // activation_grad -> linear1_grad -> layer_norm_grad
  // 2. dropout2_grad -> linear2_grad -> dropout1_grad -> activation_grad ->
  // linear1_grad -> layer_norm_grad
  // 3. layer_norm_grad -> residual_add_grad -> dropout2_grad -> linear2_grad ->
  // dropout1_grad -> activation_grad -> linear1_grad
  // 4. layer_norm_grad -> dropout2_grad -> linear2_grad -> dropout1_grad ->
  // activation_grad -> linear1_grad
  GraphPatternDetector gpd;

  auto *x_grad = gpd.mutable_pattern()
                     ->NewNode(patterns::PDNodeName(scope_name, "x_grad"))
                     ->AsInput();

  patterns::FusedFeedForwardBwd fused_feedforward_pattern(gpd.mutable_pattern(),
                                                          scope_name);
  std::unordered_set<std::string> act_grad_types = {"gelu_grad", "relu_grad"};
  fused_feedforward_pattern(
      x_grad, act_grad_types, pre_layer_norm, add_residual);

  int found_fused_feedforward_bwd_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle feed_forward backward fusion";

    // LayerNorm Grad
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_op_grad, layer_norm_op_grad, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_in, layer_norm_in, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_mean, layer_norm_mean, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_variance, layer_norm_variance, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_scale, layer_norm_scale, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_bias, layer_norm_bias, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_in_grad, layer_norm_in_grad, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_scale_grad,
                              layer_norm_scale_grad,
                              fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_bias_grad, layer_norm_bias_grad, fused_feedforward_pattern);
    // Linear Grad 1
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_op_grad_1, matmul_op_grad_1, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_in_1, matmul_in_1, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_w_1, matmul_w_1, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_in_grad_1, matmul_in_grad_1, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_w_grad_1, matmul_w_grad_1, fused_feedforward_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_op_grad_1, ele_add_op_grad_1, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_in_1, ele_add_in_1, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_bias_1, ele_add_bias_1, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_in_grad_1, ele_add_in_grad_1, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_bias_grad_1, ele_add_bias_grad_1, fused_feedforward_pattern);
    //  Dropout Grad 1
    GET_IR_NODE_FROM_SUBGRAPH(
        dropout_op_grad_1, dropout_op_grad_1, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        dropout_mask_1, dropout_mask_1, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        dropout_in_grad_1, dropout_in_grad_1, fused_feedforward_pattern);
    // Activation Grad
    GET_IR_NODE_FROM_SUBGRAPH(
        act_op_grad, act_op_grad, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act_in, act_in, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        act_in_grad, act_in_grad, fused_feedforward_pattern);
    // Linear Grad 2
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_op_grad_2, matmul_op_grad_2, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_in_2, matmul_in_2, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_w_2, matmul_w_2, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_in_grad_2, matmul_in_grad_2, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_w_grad_2, matmul_w_grad_2, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_op_grad_2, ele_add_op_grad_2, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_in_2, ele_add_in_2, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_bias_2, ele_add_bias_2, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_in_grad_2, ele_add_in_grad_2, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        ele_add_bias_grad_2, ele_add_bias_grad_2, fused_feedforward_pattern);
    //  Dropout Grad 2
    GET_IR_NODE_FROM_SUBGRAPH(
        dropout_op_grad_2, dropout_op_grad_2, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        dropout_mask_2, dropout_mask_2, fused_feedforward_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        dropout_in_grad_2, dropout_in_grad_2, fused_feedforward_pattern);

    if (PADDLE_GET_CONST(bool, dropout_op_grad_1->Op()->GetAttr("is_test")) ||
        PADDLE_GET_CONST(bool, dropout_op_grad_2->Op()->GetAttr("is_test"))) {
      LOG(WARNING)
          << "Dropout_grad 1 and dropout_grad 2 attribute is_test should "
          << "both be set false. Skip fused_feedforward_grad pattern "
             "replacement";
      return;
    }

    OpDesc fused_feedforward_op_desc(layer_norm_op_grad->Op()->Block());

    fused_feedforward_op_desc.SetType("fused_feedforward_grad");
    fused_feedforward_op_desc.SetInput(framework::GradVarName("Out"),
                                       {subgraph.at(x_grad)->Name()});
    fused_feedforward_op_desc.SetInput(
        "X", {pre_layer_norm ? layer_norm_in->Name() : matmul_in_1->Name()});
    fused_feedforward_op_desc.SetInput("Linear1Weight", {matmul_w_1->Name()});
    fused_feedforward_op_desc.SetInput("Linear1Bias", {ele_add_bias_1->Name()});
    fused_feedforward_op_desc.SetInput("Linear2Weight", {matmul_w_2->Name()});
    fused_feedforward_op_desc.SetInput("Linear2Bias", {ele_add_bias_2->Name()});
    fused_feedforward_op_desc.SetInput("Dropout1Mask",
                                       {dropout_mask_1->Name()});
    fused_feedforward_op_desc.SetInput("Dropout2Mask",
                                       {dropout_mask_2->Name()});
    fused_feedforward_op_desc.SetInput("Linear1Out", {act_in->Name()});
    fused_feedforward_op_desc.SetInput("Dropout1Out", {matmul_in_2->Name()});
    if (pre_layer_norm) {
      fused_feedforward_op_desc.SetInput("Ln1Scale",
                                         {layer_norm_scale->Name()});
      fused_feedforward_op_desc.SetInput("Ln1Bias", {layer_norm_bias->Name()});
      fused_feedforward_op_desc.SetInput("Ln1Out", {matmul_in_1->Name()});
      fused_feedforward_op_desc.SetInput("Ln1Mean", {layer_norm_mean->Name()});
      fused_feedforward_op_desc.SetInput("Ln1Variance",
                                         {layer_norm_variance->Name()});
      fused_feedforward_op_desc.SetOutput(GradVarName("Ln1Scale"),
                                          {layer_norm_scale_grad->Name()});
      fused_feedforward_op_desc.SetOutput(GradVarName("Ln1Bias"),
                                          {layer_norm_bias_grad->Name()});
    } else {
      fused_feedforward_op_desc.SetInput("Ln2Scale",
                                         {layer_norm_scale->Name()});
      fused_feedforward_op_desc.SetInput("Ln2Bias", {layer_norm_bias->Name()});
      fused_feedforward_op_desc.SetInput("Ln2Mean", {layer_norm_mean->Name()});
      fused_feedforward_op_desc.SetInput("Ln2Variance",
                                         {layer_norm_variance->Name()});
      fused_feedforward_op_desc.SetOutput(GradVarName("Ln2Scale"),
                                          {layer_norm_scale_grad->Name()});
      fused_feedforward_op_desc.SetOutput(GradVarName("Ln2Bias"),
                                          {layer_norm_bias_grad->Name()});
      // Special
      if (add_residual) {
        GET_IR_NODE_FROM_SUBGRAPH(
            ele_add_in_3, ele_add_in_3, fused_feedforward_pattern);
        fused_feedforward_op_desc.SetInput("Dropout2Out",
                                           {ele_add_in_3->Name()});
      } else {
        fused_feedforward_op_desc.SetInput("Dropout2Out",
                                           {layer_norm_in->Name()});
      }
    }

    fused_feedforward_op_desc.SetOutput(GradVarName("Linear1Weight"),
                                        {matmul_w_grad_1->Name()});
    fused_feedforward_op_desc.SetOutput(GradVarName("Linear1Bias"),
                                        {ele_add_bias_grad_1->Name()});
    fused_feedforward_op_desc.SetOutput(GradVarName("Linear2Weight"),
                                        {matmul_w_grad_2->Name()});
    fused_feedforward_op_desc.SetOutput(GradVarName("Linear2Bias"),
                                        {ele_add_bias_grad_2->Name()});
    if (add_residual) {
      GET_IR_NODE_FROM_SUBGRAPH(sum_out, sum_out, fused_feedforward_pattern);
      fused_feedforward_op_desc.SetOutput(GradVarName("X"), {sum_out->Name()});
    } else {
      if (pre_layer_norm) {
        fused_feedforward_op_desc.SetOutput(GradVarName("X"),
                                            {layer_norm_in_grad->Name()});
      } else {
        fused_feedforward_op_desc.SetOutput(GradVarName("X"),
                                            {matmul_in_grad_1->Name()});
      }
    }

    fused_feedforward_op_desc.SetAttr("pre_layer_norm", pre_layer_norm);
    fused_feedforward_op_desc.SetAttr(
        "ln1_epsilon", layer_norm_op_grad->Op()->GetAttr("epsilon"));
    fused_feedforward_op_desc.SetAttr(
        "ln2_epsilon", layer_norm_op_grad->Op()->GetAttr("epsilon"));
    fused_feedforward_op_desc.SetAttr("act_method",
                                      act_op_grad->Op()->Type().substr(0, 4));
    fused_feedforward_op_desc.SetAttr(
        "dropout1_rate", dropout_op_grad_1->Op()->GetAttr("dropout_prob"));
    fused_feedforward_op_desc.SetAttr(
        "dropout2_rate", dropout_op_grad_2->Op()->GetAttr("dropout_prob"));
    fused_feedforward_op_desc.SetAttr(
        "dropout1_implementation",
        dropout_op_grad_1->Op()->GetAttr("dropout_implementation"));
    fused_feedforward_op_desc.SetAttr(
        "dropout2_implementation",
        dropout_op_grad_2->Op()->GetAttr("dropout_implementation"));
    fused_feedforward_op_desc.SetAttr("add_residual", add_residual);
    // These attributes set default value
    fused_feedforward_op_desc.SetAttr("is_test", false);
    fused_feedforward_op_desc.SetAttr("dropout1_fix_seed", false);
    fused_feedforward_op_desc.SetAttr("dropout2_fix_seed", false);
    fused_feedforward_op_desc.SetAttr("dropout1_seed", 0);
    fused_feedforward_op_desc.SetAttr("dropout2_seed", 0);
    fused_feedforward_op_desc.SetAttr("ring_id", -1);

    auto fused_feedforward_node = g->CreateOpNode(&fused_feedforward_op_desc);

    IR_NODE_LINK_TO(subgraph.at(x_grad), fused_feedforward_node);
    IR_NODE_LINK_TO(matmul_w_1, fused_feedforward_node);
    IR_NODE_LINK_TO(ele_add_bias_1, fused_feedforward_node);
    IR_NODE_LINK_TO(matmul_w_2, fused_feedforward_node);
    IR_NODE_LINK_TO(ele_add_bias_2, fused_feedforward_node);
    IR_NODE_LINK_TO(dropout_mask_1, fused_feedforward_node);
    IR_NODE_LINK_TO(dropout_mask_2, fused_feedforward_node);
    IR_NODE_LINK_TO(act_in, fused_feedforward_node);
    IR_NODE_LINK_TO(matmul_in_2, fused_feedforward_node);
    IR_NODE_LINK_TO(layer_norm_scale, fused_feedforward_node);
    IR_NODE_LINK_TO(layer_norm_bias, fused_feedforward_node);
    IR_NODE_LINK_TO(layer_norm_mean, fused_feedforward_node);
    IR_NODE_LINK_TO(layer_norm_variance, fused_feedforward_node);
    IR_NODE_LINK_TO(layer_norm_in, fused_feedforward_node);
    if (!pre_layer_norm) {
      IR_NODE_LINK_TO(matmul_in_1, fused_feedforward_node);
    }

    IR_NODE_LINK_TO(fused_feedforward_node, layer_norm_scale_grad);
    IR_NODE_LINK_TO(fused_feedforward_node, layer_norm_bias_grad);
    IR_NODE_LINK_TO(fused_feedforward_node, matmul_w_grad_1);
    IR_NODE_LINK_TO(fused_feedforward_node, ele_add_bias_grad_1);
    IR_NODE_LINK_TO(fused_feedforward_node, matmul_w_grad_2);
    IR_NODE_LINK_TO(fused_feedforward_node, ele_add_bias_grad_2);

    if (add_residual) {
      GET_IR_NODE_FROM_SUBGRAPH(sum_out, sum_out, fused_feedforward_pattern);
      IR_NODE_LINK_TO(fused_feedforward_node, sum_out);
    } else {
      if (pre_layer_norm) {
        IR_NODE_LINK_TO(fused_feedforward_node, layer_norm_in_grad);
      } else {
        IR_NODE_LINK_TO(fused_feedforward_node, matmul_in_grad_1);
      }
    }

    std::unordered_set<const Node *> nodes_to_remove = {layer_norm_op_grad,
                                                        matmul_op_grad_1,
                                                        ele_add_op_grad_1,
                                                        dropout_op_grad_1,
                                                        act_op_grad,
                                                        matmul_op_grad_2,
                                                        ele_add_op_grad_2,
                                                        dropout_op_grad_2};

    if (add_residual) {
      GET_IR_NODE_FROM_SUBGRAPH(
          ele_add_op_grad_3, ele_add_op_grad_3, fused_feedforward_pattern);
      // Sum for gradient addition
      GET_IR_NODE_FROM_SUBGRAPH(sum_op, sum_op, fused_feedforward_pattern);
      nodes_to_remove.insert(ele_add_op_grad_3);
      nodes_to_remove.insert(sum_op);
    }
    GraphSafeRemoveNodes(g, nodes_to_remove);
    found_fused_feedforward_bwd_count++;
  };

  gpd(graph, handler);
  AddStatis(found_fused_feedforward_bwd_count);
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fused_feedforward_pass,
              paddle::framework::ir::FusedFeedForwardPass);
