/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>
#include <string>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class FusedMultiTransformerOp : public framework::OperatorWithKernel {
 private:
  static constexpr const char *OpName = "FusedAttentionOp";

 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
#define CHECK_INPUT(name) \
  OP_INOUT_CHECK(ctx->HasInput(#name), "Input", #name, OpName)
#define CHECK_INPUTS(name) \
  OP_INOUT_CHECK(ctx->HasInputs(#name), "Input", #name, OpName)
#define CHECK_OUTPUT(name) \
  OP_INOUT_CHECK(ctx->HasOutput(#name), "Output", #name, OpName)
#define CHECK_OUTPUTS(name) \
  OP_INOUT_CHECK(ctx->HasOutputs(#name), "Output", #name, OpName)

    CHECK_INPUT(X);

    // attention
    CHECK_INPUTS(QKVW);
    CHECK_INPUTS(OutLinearW);

    if (ctx->HasInputs("CacheKV")) {
      CHECK_OUTPUTS(CacheKVOut);
    }

    // ffn
    CHECK_INPUTS(FFN1Weight);
    CHECK_INPUTS(FFN2Weight);

    CHECK_OUTPUT(Out);

    // x: qkv's input [batch_size, seq_len, dim_embed]
    // y: qkv's weight: [3, num_head, dim_head, dim_embed]
    auto x_dim = ctx->GetInputDim("X");
    auto y_dim = ctx->GetInputsDim("QKVW")[0];
    PADDLE_ENFORCE_EQ(x_dim.size(), 3, platform::errors::InvalidArgument(
                                           "The dimensions of x must be 3"
                                           "(batch_size, seq_len, dim_embed),"
                                           "but received dimensions of"
                                           "Input is [%d]",
                                           x_dim.size()));
    PADDLE_ENFORCE_EQ(y_dim.size(), 4,
                      platform::errors::InvalidArgument(
                          "The dimensions of qkv_weight must be 4"
                          "(3, num_head, dim_head, dim_embed),"
                          "but received dimensions of"
                          "Input is [%d]",
                          y_dim.size()));
    PADDLE_ENFORCE_EQ(x_dim[2], y_dim[3],
                      platform::errors::InvalidArgument(
                          "ShapeError: the dimension of x_dim[2] and y_dim[3]"
                          "must be equal. But received: the shape "
                          "of input x = [%s], and the shape of "
                          "input qkv_weight = [%s]",
                          x_dim, y_dim));

    if (ctx->Attrs().Get<int>("ring_id") == -1) {
      PADDLE_ENFORCE_EQ(y_dim[1] * y_dim[2], y_dim[3],
                        platform::errors::InvalidArgument(
                            "The dimensions of qkv_weight must be 4"
                            "(3, num_head, dim_head, dim_embed),"
                            "and must satisfy the limitations: "
                            "(num_head * dim_head == dim_embed)"));
    }

    // cache_seq_len + seq_len if cache else seq_len
    auto out_seq_len = x_dim[1];
    if (ctx->HasInputs("CacheKV")) {
      // [2, batch_size, num_head, cache_seq_len, head_size]
      const auto &c_dims = ctx->GetInputsDim("CacheKV");
      const auto &c_dim = c_dims[0];

      PADDLE_ENFORCE_EQ(
          c_dim.size(), 5,
          paddle::platform::errors::InvalidArgument(
              "The CacheKV must be 5 dims, but got %d", c_dim.size()));
      PADDLE_ENFORCE_EQ(c_dim[0], 2,
                        paddle::platform::errors::InvalidArgument(
                            "The first dim of CacheKV must be 2, but got %d",
                            c_dim[0]));  // 2
      PADDLE_ENFORCE_EQ(c_dim[1], x_dim[0],
                        paddle::platform::errors::InvalidArgument(
                            "The second dim of CacheKV must be equal with "
                            "batch size %d, but got %d",
                            x_dim[0], c_dim[1]));  // batch_size
      PADDLE_ENFORCE_EQ(c_dim[2], y_dim[1],
                        paddle::platform::errors::InvalidArgument(
                            "The third dim of CacheKV must be equal with num "
                            "head %d, but got %d",
                            y_dim[1], c_dim[2]));  // num_head
      PADDLE_ENFORCE_GE(
          c_dim[3], 0,
          paddle::platform::errors::InvalidArgument(
              "The forth dim of CacheKV must be greater than 0, but got %d",
              c_dim[3]));  // cache_seq_len
      PADDLE_ENFORCE_EQ(c_dim[4], y_dim[2],
                        paddle::platform::errors::InvalidArgument(
                            "The fifth dim of CacheKV must be equal with head "
                            "size %d, but got %d",
                            y_dim[2], c_dim[4]));  // head_size

      out_seq_len += c_dim[3];

      auto out_dims = c_dims;
      for (auto &out_dim : out_dims) {
        out_dim[3] = out_seq_len;
      }
      // [2, batch_size, num_head, cache_seq_len + seq_len, head_size]
      ctx->SetOutputsDim("CacheKVOut", out_dims);
    }

    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class FusedMultiTransformerOpOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor.");
    AddInput("LnScale",
             "Scale is a 1-dimensional tensor of size "
             "H. Here, H represents the last dimension of its input tensor.")
        .AsDuplicable();
    AddInput("LnBias",
             "Bias is a 1-dimensional tensor of size "
             "H. Here, H represents the last dimension of its input tensor.")
        .AsDuplicable();
    AddInput("QKVW", "The qkv weight tensor.").AsDuplicable();
    AddInput("QKVBias", "The qkv bias tensor.").AsDispensable().AsDuplicable();
    AddInput("CacheKV", "(optional) The cached KV for generation inference.")
        .AsDispensable()
        .AsDuplicable();
    AddInput("SrcMask", "(optional) The attention mask tensor in fmha.")
        .AsDispensable();
    AddInput("OutLinearW", "The out_linear weight tensor.").AsDuplicable();
    AddInput("OutLinearBias", "The out_linear bias tensor.")
        .AsDispensable()
        .AsDuplicable();

    AddInput("FFNLnScale", "The layer_norm scale of FusedFeedForward op")
        .AsDuplicable();
    AddInput("FFNLnBias", "The layer_norm bias of FusedFeedForward op")
        .AsDuplicable();
    AddInput("FFN1Weight", "The linear1 weight of FusedFeedForward op")
        .AsDuplicable();
    AddInput("FFN1Bias", "The linear1 bias of FusedFeedForward op")
        .AsDispensable()
        .AsDuplicable();
    AddInput("FFN2Weight", "The linear2 weight of FusedFeedForward op")
        .AsDuplicable();
    AddInput("FFN2Bias", "The linear2 bias input of FusedFeedForward op")
        .AsDispensable()
        .AsDuplicable();

    AddOutput("CacheKVOut", "The udpated cache KV.").AsDuplicable();
    AddOutput("Out", "Result after multi .");

    AddAttr<bool>("pre_layer_norm",
                  "if true, the attention op uses pre_layer_norm architecure, "
                  "else, uses post_layer_norm architecuture. "
                  "[default true].")
        .SetDefault(true);
    AddAttr<float>("epsilon",
                   "Constant for numerical stability [default 1e-5].")
        .SetDefault(1e-5)
        .AddCustomChecker([](const float &epsilon) {
          PADDLE_ENFORCE_EQ(epsilon >= 0.0f && epsilon <= 0.001f, true,
                            platform::errors::InvalidArgument(
                                "'epsilon' in Op(LayerNorm) should be between"
                                "0.0 and 0.001, But received [%s].",
                                epsilon));
        });

    // for dropout in fmha.
    AddAttr<float>("attn_dropout_rate", "Probability of setting units to zero.")
        .SetDefault(.5f)
        .AddCustomChecker([](const float &drop_p) {
          PADDLE_ENFORCE_EQ(
              drop_p >= 0.0f && drop_p <= 1.0f, true,
              platform::errors::InvalidArgument(
                  "'attn_dropout_rate' must be between 0.0 and 1.0."));
        });
    AddAttr<bool>("attn_dropout_is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddAttr<std::string>(
        "attn_dropout_implementation",
        "[\"downgrade_in_infer\"|\"upscale_in_train\"]"
        "There are two kinds of ways to implement dropout"
        "(the mask below is a tensor have the same shape with input"
        "the value of mask is 0 or 1, the ratio of 0 is dropout_rate)"
        "1. downgrade_in_infer(default), downgrade the outcome at inference "
        "time"
        "   train: out = input * mask"
        "   inference: out = input * (1.0 - dropout_rate)"
        "2. upscale_in_train, upscale the outcome at training time, do nothing "
        "in inference"
        "   train: out = input * mask / ( 1.0 - dropout_rate )"
        "   inference: out = input"
        "   dropout op can be removed from the program. the program will be "
        "efficient")
        .SetDefault("upscale_in_train")
        .AddCustomChecker([](const std::string &type) {
          PADDLE_ENFORCE_EQ(
              type == "downgrade_in_infer" || type == "upscale_in_train", true,
              platform::errors::InvalidArgument(
                  "dropout_implementation can only be downgrade_in_infer or "
                  "upscale_in_train"));
        });

    AddAttr<float>("dropout_rate", "Probability of setting units to zero.")
        .SetDefault(.5f)
        .AddCustomChecker([](const float &drop_p) {
          PADDLE_ENFORCE_EQ(drop_p >= 0.0f && drop_p <= 1.0f, true,
                            platform::errors::InvalidArgument(
                                "'dropout_rate' must be between 0.0 and 1.0."));
        });

    AddAttr<bool>("dropout_is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddAttr<std::string>(
        "dropout_implementation",
        "[\"downgrade_in_infer\"|\"upscale_in_train\"]"
        "The meaning is the same as 'attn_dropout_implementation'.")
        .SetDefault("downgrade_in_infer")
        .AddCustomChecker([](const std::string &type) {
          PADDLE_ENFORCE_EQ(
              type == "downgrade_in_infer" || type == "upscale_in_train", true,
              platform::errors::InvalidArgument(
                  "dropout_implementation can only be downgrade_in_infer or "
                  "upscale_in_train"));
        });
    AddAttr<std::string>("act_method", "act_method").SetDefault("gelu");

    AddAttr<int>(
        "ring_id",
        "ring id for tensor model parallel. distributed training and inference")
        .SetDefault(-1);

    AddComment(R"DOC(
  Add fused attention op whose logic is as follows:
  // @input: [batch_size, seq_len, 3, num_head, head_dim] 
  // @final_out: [batch_size, seq_len, num_heads, head_dim] 
  if (pre_layernorm)
    out = layer_norm(input);
	out = compute_qkv(out) + bias;
	// fmha module
  {
    out = transpose(out, perm=[2, 0, 3, 1, 4]);
    out = q * k^t;
    out = attn_mask + out;
    out = softmax(out);
    out = dropout(out);
    out = out * v;
    out = transpose(out, perm=[0, 2, 1, 3]);
                
  }
	out = out_linear(out);
  if (pre_layernorm)
    final_out = residual + dropout(bias + out);
  else
    final_out = layer_norm(residual + dropout(bias + out));
    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    fused_multi_transformer, ops::FusedMultiTransformerOp,
    ops::FusedMultiTransformerOpOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
