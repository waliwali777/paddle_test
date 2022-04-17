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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/ir/mkldnn/quant_dequant_mkldnn_v2_fuse_pass.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/ir/mkldnn/mkldnn_pass_util.h"

namespace paddle {
namespace framework {
namespace ir {

void QuantDequantMkldnnV2FusePass::ApplyImpl(ir::Graph* graph) const {
  VLOG(0) << "Convert dynamic graph to INT8 mkldnn model";

  FusePassBase::Init("quant_dequant_mkldnn_v2_fuse_pass", graph);
  GraphPatternDetector gpd;

  std::unordered_set<std::string> quantize_linear_types = {"quantize_linear"};
  std::unordered_set<std::string> dequantize_linear_types = {
      "dequantize_linear"};

  std::unordered_map<std::string, std::vector<float>> weights_thresholds{};
  std::unordered_map<std::string, std::vector<float>> var_quant_scales{};

  auto* scope = param_scope();

  GatherInputWeightsScalesFromFake(graph, scope, quantize_linear_types,
                                   dequantize_linear_types, &weights_thresholds,
                                   &var_quant_scales);
  RemoveQuantDequantLinearOps(graph, scope);
  
  RemoveDequantLinearOps(graph, scope);
  
  SaveInfoInTheFirstOp(graph, "has_quant_info", "var_quant_scales",
                       var_quant_scales);
}

void QuantDequantMkldnnV2FusePass::GatherInputWeightsScalesFromFake(
    ir::Graph* graph, Scope* scope,
    std::unordered_set<std::string> quantize_linear_types,
    std::unordered_set<std::string> dequantize_linear_types,
    std::unordered_map<std::string, std::vector<float>>* weights_thresholds,
    std::unordered_map<std::string, std::vector<float>>* var_quant_scales)
    const {
  VLOG(0) << "Gather input and weight scales from dequantize_linear";

  for (auto* op_node :
       ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp()) continue;

    if (dequantize_linear_types.find(op_node->Name()) !=
        dequantize_linear_types.end()) {
      auto* op_desc = op_node->Op();
      const int bit_length =
          BOOST_GET_CONST(int, op_desc->GetAttr("bit_length"));
      const int quant_axis =
          BOOST_GET_CONST(int, op_desc->GetAttr("quant_axis"));
      auto& y_var_name = op_desc->Output("Y")[0];

      PADDLE_ENFORCE_EQ(bit_length, 8, platform::errors::InvalidArgument(
                                           "Unsupported number quantization "
                                           "bits: %d, only 8 is supported now.",
                                           bit_length));

      if (var_quant_scales->find(y_var_name) == var_quant_scales->end()) {
        
        VLOG(1)<<"y_var_name: "<< y_var_name;
        
        auto& zeropoint_var_name = op_desc->Input("ZeroPoint")[0];
        auto* zeropoint_var = scope->FindVar(zeropoint_var_name);
        PADDLE_ENFORCE_NOT_NULL(
            zeropoint_var, "The zeropoint_var is not found.");	
        auto* zeropoint_tensor = zeropoint_var->GetMutable<LoDTensor>();

        auto zeropoint_data =
            zeropoint_tensor->mutable_data<int>(platform::CPUPlace());

        auto& scale_var_name = op_desc->Input("Scale")[0];
        auto* scale_var = scope->FindVar(scale_var_name);
        PADDLE_ENFORCE_NOT_NULL(
            scale_var, "The scale_var is not found.");	
        auto* scale_tensor = scale_var->GetMutable<LoDTensor>();
        auto scale_data =
            scale_tensor->mutable_data<float>(platform::CPUPlace());
        PADDLE_ENFORCE_EQ(scale_tensor->numel(), zeropoint_tensor->numel(), platform::errors::InvalidArgument(
                                                "scale vec should be same size as zeropoint vec"
                                                "but now scale size is: %d, zeropoint size is: %d.",
                                                scale_tensor->numel(), zeropoint_tensor->numel()));
              
        if (quant_axis < 0 && scale_tensor->numel() != 1 ){
          std::cout<<"This is big error"<<std::endl;
        } 
        
        size_t scale_zeropoint_size = scale_tensor->numel();
        // std::vector<float> scale_zero_vec(scale_zeropoint_size*2, 0.0f);
        std::vector<float> scale_zero_vec(scale_zeropoint_size, 0.0f);
        
        VLOG(1)<<"zeropoint_data: ";
        
        for (size_t i = 0; i < scale_zeropoint_size; i++){
          scale_zero_vec[i] = static_cast<float>(zeropoint_data[i]);
          VLOG(1)<<scale_zero_vec[i]<<" ";
        }  
        
        VLOG(1)<<"scale_data: ";
        // for (size_t i = 0; i < scale_zeropoint_size; i++){
        //   scale_zero_vec[i + scale_zeropoint_size] = scale_data[i];
        //   VLOG(0)<<scale_zero_vec[i + scale_zeropoint_size] <<" ";
        // }       

        for (size_t i = 0; i < scale_zeropoint_size; i++){
          scale_zero_vec[i] = 1.0 / scale_data[i];
          VLOG(1)<<scale_zero_vec[i] <<" ";
        }
        var_quant_scales->insert(std::make_pair(y_var_name, scale_zero_vec));
      }
    }
  }
}

void QuantDequantMkldnnV2FusePass::RemoveQuantDequantLinearOps(
    ir::Graph* graph, Scope* scope) const {
  VLOG(0) << "Fuse quantize_linear->dequantize_linear ops";
  GraphPatternDetector gpd;
  patterns::QuantizeDequantizeLinearPattern qdq_pattern(gpd.mutable_pattern(),
                                                        qdq_name_scope_);
  qdq_pattern();
  int found_quantize_dequantize_linear_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(1) << "handle quantize dequantize linear fuse pass";
    GET_IR_NODE_FROM_SUBGRAPH(prev_op, prev_op, qdq_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quantize_linear_in_x, quantize_linear_in_x,
                              qdq_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quantize_linear_in_scale,
                              quantize_linear_in_scale, qdq_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quantize_linear_in_zeropoint,
                              quantize_linear_in_zeropoint, qdq_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quantize_linear_op, quantize_linear_op,
                              qdq_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quantize_linear_out, quantize_linear_out,
                              qdq_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequantize_linear_op, dequantize_linear_op,
                              qdq_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequantize_linear_out, dequantize_linear_out,
                              qdq_pattern);

    prev_op->Op()->RenameOutput(quantize_linear_in_x->Name(), dequantize_linear_out->Name());
    
    IR_NODE_LINK_TO(prev_op, dequantize_linear_out);
    GraphSafeRemoveNodes(graph,
                         {quantize_linear_in_x, quantize_linear_in_scale,
                          quantize_linear_in_zeropoint, quantize_linear_op,
                          quantize_linear_out, dequantize_linear_op});

    found_quantize_dequantize_linear_count++;
  };
  gpd(graph, handler);
  AddStatis(found_quantize_dequantize_linear_count);
}

template <typename T = float>
void DequantizeV2Weights(const framework::Tensor* in, const framework::Tensor* scale,
                  T max_range, const int quant_axis, framework::Tensor* out){
    auto in_dims=in->dims();
    const int64_t channel = in_dims[quant_axis];
    const T* scale_factor = scale->data<T>();
    if (quant_axis == 0) {
      auto* in_data = in->data<int8_t>();
      auto* out_data = out->mutable_data<float>(platform::CPUPlace());
      auto single_scale_nums = in->numel() / in->dims()[0];
      for (int64_t i = 0; i < channel; i++) {
        T s = scale_factor[i];
        VLOG(1)<<"After getting s data";
        for (auto j = 0; j < single_scale_nums ; j++){
          *(out_data + i * single_scale_nums + j) = *(in_data + i * single_scale_nums + j) * s / max_range;
        }
      }
    } else if (quant_axis == 1) {
      int64_t out_iter = 1;
      for (int i = 0; i < quant_axis; i++) {
        out_iter *= in_dims[i];
      }
      int64_t step_i = in->numel() / out_iter;
      int64_t step_j = in->numel() / (out_iter * channel);
      auto* in_data = in->data<int8_t>();
      auto* out_data = out->mutable_data<T>(platform::CPUPlace());
      for (int64_t i = 0; i < out_iter; i++) {
        for (int64_t j = 0; j < channel; j++) {
          auto* cur_in = in_data + i * step_i + j * step_j;
          auto* cur_out = out_data + i * step_i + j * step_j;
          T s = scale_factor[j];
          for (int64_t k = 0; k < step_j; k++) {
            *cur_out = (*cur_in) * s / max_range;
            ++cur_in;
            ++cur_out;
          }
        }
      }
    }
  }

void QuantDequantMkldnnV2FusePass::RemoveDequantLinearOps(
    ir::Graph* graph, Scope* scope) const {
  GraphPatternDetector gpd;
  patterns::DequantizeLinearPattern dq_pattern(gpd.mutable_pattern(),
                                                        dq_name_scope_);
  dq_pattern();                                                          
  int found_dequantize_linear_count = 0;
  VLOG(1) << "handle removing dequantize_linear pass";
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(1) << "handle dequantize_linear removing pass";
    GET_IR_NODE_FROM_SUBGRAPH(dequantize_linear_in_x, dequantize_linear_in_x,
                              dq_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequantize_linear_in_scale,
                              dequantize_linear_in_scale, dq_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequantize_linear_in_zeropoint,
                              dequantize_linear_in_zeropoint, dq_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequantize_linear_op, dequantize_linear_op,
                              dq_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequantize_linear_out, dequantize_linear_out,
                              dq_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(next_op, next_op,
                              dq_pattern);

    auto* dequantize_op_desc = dequantize_linear_op->Op();
    const int bit_length =
          BOOST_GET_CONST(int, dequantize_op_desc->GetAttr("bit_length"));
    const int quant_axis =
          BOOST_GET_CONST(int, dequantize_op_desc->GetAttr("quant_axis"));
    // get the in weights int8 tensor
    auto* dequant_in_x_var = scope->FindVar(dequantize_linear_in_x->Name());
    PADDLE_ENFORCE_NOT_NULL(
            dequant_in_x_var, "The scale_var is not found.");	
    auto* dequant_in_x_tensor = dequant_in_x_var->GetMutable<LoDTensor>();
    VLOG(1) << "dequant_in_x_tensor weights are: "<<(*dequant_in_x_tensor);    

    // get the scale tensor
    auto* dequant_in_scale_var = scope->FindVar(dequantize_linear_in_scale->Name());
    PADDLE_ENFORCE_NOT_NULL(
            dequant_in_scale_var, "The dequant_in_scale_var is not found.");	
    auto* dequant_in_scale_tensor = dequant_in_scale_var->GetMutable<LoDTensor>();

    // Force quant_axis aligned with scale size
    PADDLE_ENFORCE_EQ(dequant_in_scale_tensor->numel(), dequant_in_x_tensor->dims()[quant_axis],
    platform::errors::PreconditionNotMet(
        "The number of first scale values must be the same with "
        "quant_axis dimension value of Input(X) when the `scale` has "
        "only one element, but %ld != %ld here.",
        dequant_in_scale_tensor->numel(), dequant_in_x_tensor->dims()[quant_axis]));
    auto max_range = (std::pow(2, bit_length - 1) - 1);
    VLOG(1)<<"max_range is: "<<max_range;

    // Create output node and var
    auto* dequantize_linear_out_var = dequantize_linear_out->Var();
    dequantize_linear_out_var->SetShape(phi::vectorize(dequant_in_x_tensor->dims()));
    dequantize_linear_out_var->SetDataType(proto::VarType::FP32);
    dequantize_linear_out_var->SetLoDLevel(dequant_in_x_tensor->lod().size());
    dequantize_linear_out_var->SetPersistable(true);
    auto* dequantize_weights_node = graph->CreateVarNode(dequantize_linear_out_var);
    auto* dequantize_weights_tensor = scope->Var(dequantize_weights_node->Name())->GetMutable<LoDTensor>();
    dequantize_weights_tensor->Resize(dequant_in_x_tensor->dims());

    std::fill_n(dequantize_weights_tensor->mutable_data<float>(platform::CPUPlace()),
                dequant_in_x_tensor->numel(), 0.0f);
    
    DequantizeV2Weights<float>(dequant_in_x_tensor, dequant_in_scale_tensor, static_cast<float>(max_range), quant_axis, dequantize_weights_tensor);

    next_op->Op()->RenameInput(dequantize_linear_out->Name(), dequantize_weights_node->Name());

    IR_NODE_LINK_TO(dequantize_weights_node, next_op);

    GraphSafeRemoveNodes(graph,
                         {dequantize_linear_in_x, dequantize_linear_in_scale,
                          dequantize_linear_in_zeropoint, dequantize_linear_op, dequantize_linear_out});
 
    found_dequantize_linear_count++;
  };

  gpd(graph, handler);
  AddStatis(found_dequantize_linear_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(quant_dequant_mkldnn_v2_fuse_pass,
              paddle::framework::ir::QuantDequantMkldnnV2FusePass);

REGISTER_PASS_CAPABILITY(quant_dequant_mkldnn_v2_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("quantize_linear", 0)
            .EQ("dequantize_linear", 0));
