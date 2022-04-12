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

#include "paddle/fluid/framework/ir/mkldnn/params_to_int8_pass.h"

#include "paddle/fluid/framework/op_version_registry.h"
// #include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

ParamsToInt8Pass::ParamsToInt8Pass() {
  AddOpCompat(OpCompat("conv2d"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("Filter")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .IsOptional()
      .End()
      .AddInput("ResidualData")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Output")
      .IsTensor()
      .End()
      .AddAttr("strides")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("paddings")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("padding_algorithm")
      .IsOptional()
      .IsStringIn({"EXPLICIT", "SAME", "VALID"})
      .End()
      .AddAttr("groups")
      .IsNumGE(1)
      .End()
      .AddAttr("dilations")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("data_format")
      .IsStringIn({"NCHW", "AnyLayout"})
      .End();
}

void ParamsToInt8Pass::Conv(ir::Graph* graph) const {
  std::string name_scope = "params_to_int8_pass";
  GraphPatternDetector gpd;
  patterns::Conv conv_pattern(gpd.mutable_pattern(), name_scope);
  conv_pattern();

  int params_to_int8_conv_found = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    VLOG(4) << "handle convolution params_to_int8_pass";

    GET_IR_NODE_FROM_SUBGRAPH(conv_op, conv_op, conv_pattern);

    if (!platform::HasOpINT8DataType(conv_op->Op())) {
      return;
    }

    GET_IR_NODE_FROM_SUBGRAPH(conv_input, conv_input, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_filter, conv_filter, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_output, conv_output, conv_pattern);

    auto input_names = conv_op->Op()->InputNames();
    bool has_bias = std::find(input_names.begin(), input_names.end(), "Bias") !=
                    input_names.end();
    std::vector<int64_t> weights_tz = conv_filter->Var()->GetShape();
    const int groups =
        std::max(conv_op->Op()->GetAttrIfExists<int>("groups"), 1);

    const auto& scale_weights_data =
        conv_op->Op()->GetAttrIfExists<std::vector<float>>("Scale_weights");

    bool is_multi_channel = scale_weights_data.size() > 1;

    int count = 1;
    if (is_multi_channel) {
      count *= weights_tz[0];
      if (groups > 1) {
        count *= weights_tz[1];
      }
    }

    // get scope to interact with tensors
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, platform::errors::InvalidArgument("Scope cannot be nullptr."));

    // get float weights
    auto* weights =
        scope->FindVar(conv_filter->Name())->GetMutable<LoDTensor>();

    if (weights->dtype() != phi::DataType::FLOAT32) {
      VLOG(4) << "Skipping convolution (id: " << conv_op->id()
              << ") because it's a bug that it is detected again.";
      return;
    }

    auto weights_shape = weights->dims();
    float* weights_data = weights->data<float>();

    // Create int8 weights variable
    VarDesc int_weights(
        patterns::PDNodeName(name_scope, "conv2d_int8_weights"));
    int_weights.SetShape(weights_tz);
    int_weights.SetDataType(framework::proto::VarType::Type::VarType_Type_INT8);
    // int_weights.SetLoDLevel(conv_filter->Var()->GetLoDLevel());
    int_weights.SetPersistable(true);
    ir::Node* int_weights_node = g->CreateVarNode(&int_weights);
    auto* int_weights_tensor =
        scope->Var(int_weights_node->Name())->GetMutable<LoDTensor>();

    // Initialize int8 weights
    int_weights_tensor->Resize(weights_shape);
    auto int_weights_data =
        int_weights_tensor->mutable_data<int8_t>(platform::CPUPlace());
    // Quantize weights to int8
    auto len = weights->numel() / count;
    if (is_multi_channel) {
      for (int i = 0; i < count; ++i) {
        auto scale = scale_weights_data[i];
        int_weights_data =
            std::transform(weights_data, weights_data + len, int_weights_data,
                           [&scale](float f) {
                             return static_cast<int8_t>(std::round(f * scale));
                           });
        weights_data += len;
      }
    } else {
      auto len = int_weights_tensor->numel();
      auto scale = scale_weights_data[0];
      std::transform(weights_data, weights_data + len, int_weights_data,
                     [&scale](float f) {
                       return static_cast<int8_t>(std::round(f * scale));
                     });
    }

    // connect new int8 tensor to weights input
    conv_op->Op()->SetInput("Filter", {int_weights_node->Name()});
    IR_NODE_LINK_TO(int_weights_node, conv_op);
    GraphSafeRemoveNodes(graph, {conv_filter});
    conv_op->Op()->SetAttr("Scale_weights", std::vector<float>(1, 1));

    // Get float biases
    if (has_bias && conv_op->Op()->Input("Bias").size() > 0) {
      std::string conv_bias_name = conv_op->Op()->Input("Bias")[0];
      auto bias_tensor =
          scope->FindVar(conv_bias_name)->GetMutable<LoDTensor>();
      PADDLE_ENFORCE_EQ(count, bias_tensor->numel());

      // Create int32 biases variable
      VarDesc int_biases(
          patterns::PDNodeName(name_scope, "conv2d_int32_biases"));
      int_biases.SetShape(phi::vectorize(bias_tensor->dims()));
      int_biases.SetDataType(
          framework::proto::VarType::Type::VarType_Type_INT32);
      // int_biases.SetLoDLevel(bias_tensor->lod());
      int_biases.SetPersistable(true);
      ir::Node* int_biases_node = g->CreateVarNode(&int_biases);
      auto* int_biases_tensor =
          scope->Var(int_biases_node->Name())->GetMutable<LoDTensor>();

      // Initialize int biases
      int_biases_tensor->Resize(bias_tensor->dims());
      auto int_bias_data =
          int_biases_tensor->mutable_data<int32_t>(platform::CPUPlace());
      // Quantize biases to int32
      const auto& bias_data =
          bias_tensor->mutable_data<float>(platform::CPUPlace());
      const auto& scale_bias_data =
          conv_op->Op()->GetAttrIfExists<std::vector<float>>("Bias_scales");
      for (int i = 0; i < count; ++i) {
        int_bias_data[i] =
            static_cast<int32_t>(std::round(bias_data[i] * scale_bias_data[i]));
      }
      auto conv_bias_it =
          std::find_if(conv_op->inputs.begin(), conv_op->inputs.end(),
                       [&conv_bias_name](ir::Node* node) {
                         return node->Name() == conv_bias_name;
                       });
      PADDLE_ENFORCE_NE(
          conv_bias_it, conv_op->inputs.end(),
          platform::errors::InvalidArgument("No bias node found."));
      ir::Node* conv_bias = (*conv_bias_it);

      // connect new int32 bias to weights input
      conv_op->Op()->SetInput("Bias", {int_biases_node->Name()});
      IR_NODE_LINK_TO(int_biases_node, conv_op);
      GraphSafeRemoveNodes(graph, {conv_bias});
      conv_op->Op()->SetAttr("Bias_scales", std::vector<float>(1, 1));
    }
    params_to_int8_conv_found++;
  };
  gpd(graph, handler);
  AddStatis(params_to_int8_conv_found);

  std::stringstream msg_ss;
  msg_ss << "Quantized weights of " << params_to_int8_conv_found
         << " conv2d ops";
  paddle::string::PrettyLogDetail(msg_ss.str().c_str());
}

void ParamsToInt8Pass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          platform::errors::InvalidArgument(
                              "Pointer to graph argument should not be NULL."));
  FusePassBase::Init("params_to_int8_pass", graph);
  Conv(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(params_to_int8_pass, paddle::framework::ir::ParamsToInt8Pass);
REGISTER_PASS_CAPABILITY(params_to_int8_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().LE(
            "conv2d", 1));
