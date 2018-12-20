/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/operators/yolov3_loss_op.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class Yolov3LossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of Yolov3LossOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("GTBox"),
                   "Input(GTBox) of Yolov3LossOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("GTLabel"),
                   "Input(GTLabel) of Yolov3LossOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Loss"),
                   "Output(Loss) of Yolov3LossOp should not be null.");

    auto dim_x = ctx->GetInputDim("X");
    auto dim_gtbox = ctx->GetInputDim("GTBox");
    auto dim_gtlabel = ctx->GetInputDim("GTLabel");
    auto anchors = ctx->Attrs().Get<std::vector<int>>("anchors");
    int anchor_num = anchors.size() / 2;
    auto anchor_mask = ctx->Attrs().Get<std::vector<int>>("anchor_mask");
    int mask_num = anchor_mask.size();
    auto class_num = ctx->Attrs().Get<int>("class_num");
    PADDLE_ENFORCE_EQ(dim_x.size(), 4, "Input(X) should be a 4-D tensor.");
    PADDLE_ENFORCE_EQ(dim_x[2], dim_x[3],
                      "Input(X) dim[3] and dim[4] should be euqal.");
    PADDLE_ENFORCE_EQ(
        dim_x[1], mask_num * (5 + class_num),
        "Input(X) dim[1] should be equal to (anchor_mask_number * (5 "
        "+ class_num)).");
    PADDLE_ENFORCE_EQ(dim_gtbox.size(), 3,
                      "Input(GTBox) should be a 3-D tensor");
    PADDLE_ENFORCE_EQ(dim_gtbox[2], 4, "Input(GTBox) dim[2] should be 5");
    PADDLE_ENFORCE_EQ(dim_gtlabel.size(), 2,
                      "Input(GTBox) should be a 2-D tensor");
    PADDLE_ENFORCE_EQ(dim_gtlabel[0], dim_gtbox[0],
                      "Input(GTBox) and Input(GTLabel) dim[0] should be same");
    PADDLE_ENFORCE_EQ(dim_gtlabel[1], dim_gtbox[1],
                      "Input(GTBox) and Input(GTLabel) dim[1] should be same");
    PADDLE_ENFORCE_GT(anchors.size(), 0,
                      "Attr(anchors) length should be greater then 0.");
    PADDLE_ENFORCE_EQ(anchors.size() % 2, 0,
                      "Attr(anchors) length should be even integer.");
    for (size_t i = 0; i < anchor_mask.size(); i++) {
      PADDLE_ENFORCE_LT(
          anchor_mask[i], anchor_num,
          "Attr(anchor_mask) should not crossover Attr(anchors).");
    }
    PADDLE_ENFORCE_GT(class_num, 0,
                      "Attr(class_num) should be an integer greater then 0.");

    std::vector<int64_t> dim_out({dim_x[0]});
    ctx->SetOutputDim("Loss", framework::make_ddim(dim_out));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   platform::CPUPlace());
  }
};

class Yolov3LossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor of YOLOv3 loss operator, "
             "This is a 4-D tensor with shape of [N, C, H, W]."
             "H and W should be same, and the second dimention(C) stores"
             "box locations, confidence score and classification one-hot"
             "key of each anchor box");
    AddInput("GTBox",
             "The input tensor of ground truth boxes, "
             "This is a 3-D tensor with shape of [N, max_box_num, 5], "
             "max_box_num is the max number of boxes in each image, "
             "In the third dimention, stores x, y, w, h coordinates, "
             "x, y is the center cordinate of boxes and w, h is the "
             "width and height and x, y, w, h should be divided by "
             "input image height to scale to [0, 1].");
    AddInput("GTLabel",
             "The input tensor of ground truth label, "
             "This is a 2-D tensor with shape of [N, max_box_num], "
             "and each element shoudl be an integer to indicate the "
             "box class id.");
    AddOutput("Loss",
              "The output yolov3 loss tensor, "
              "This is a 1-D tensor with shape of [N]");

    AddAttr<int>("class_num", "The number of classes to predict.");
    AddAttr<std::vector<int>>("anchors",
                              "The anchor width and height, "
                              "it will be parsed pair by pair.")
        .SetDefault(std::vector<int>{});
    AddAttr<std::vector<int>>("anchor_mask",
                              "The mask index of anchors used in "
                              "current YOLOv3 loss calculation.")
        .SetDefault(std::vector<int>{});
    AddAttr<int>("downsample",
                 "The downsample ratio from network input to YOLOv3 loss "
                 "input, so 32, 16, 8 should be set for the first, second, "
                 "and thrid YOLOv3 loss operators.")
        .SetDefault(32);
    AddAttr<float>("ignore_thresh",
                   "The ignore threshold to ignore confidence loss.")
        .SetDefault(0.7);
    AddComment(R"DOC(
         This operator generate yolov3 loss by given predict result and ground
         truth boxes.
         
         The output of previous network is in shape [N, C, H, W], while H and W
         should be the same, specify the grid size, each grid point predict given
         number boxes, this given number is specified by anchors, it should be 
         half anchors length, which following will be represented as S. In the 
         second dimention(the channel dimention), C should be S * (class_num + 5),
         class_num is the box categoriy number of source dataset(such as coco), 
         so in the second dimention, stores 4 box location coordinates x, y, w, h 
         and confidence score of the box and class one-hot key of each anchor box.

         While the 4 location coordinates if $$tx, ty, tw, th$$, the box predictions
         correspnd to:

         $$
         b_x = \sigma(t_x) + c_x
         b_y = \sigma(t_y) + c_y
         b_w = p_w e^{t_w}
         b_h = p_h e^{t_h}
         $$

         While $$c_x, c_y$$ is the left top corner of current grid and $$p_w, p_h$$
         is specified by anchors.

         As for confidence score, it is the logistic regression value of IoU between
         anchor boxes and ground truth boxes, the score of the anchor box which has 
         the max IoU should be 1, and if the anchor box has IoU bigger then ignore 
         thresh, the confidence score loss of this anchor box will be ignored.

         Therefore, the yolov3 loss consist of three major parts, box location loss,
         confidence score loss, and classification loss. The L1 loss is used for 
         box coordinates (w, h), and sigmoid cross entropy loss is used for box 
         coordinates (x, y), confidence score loss and classification loss.

         In order to trade off box coordinate losses between big boxes and small 
         boxes, box coordinate losses will be mutiplied by scale weight, which is
         calculated as follow.

         $$
         weight_{box} = 2.0 - t_w * t_h
         $$

         Final loss will be represented as follow.

         $$
         loss = (loss_{xy} + loss_{wh}) * weight_{box}
              + loss_{conf} + loss_{class}
         $$
         )DOC");
  }
};

class Yolov3LossOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Loss")),
                   "Input(Loss@GRAD) should not be null");
    auto dim_x = ctx->GetInputDim("X");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), dim_x);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   platform::CPUPlace());
  }
};

class Yolov3LossGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto* op = new framework::OpDesc();
    op->SetType("yolov3_loss_grad");
    op->SetInput("X", Input("X"));
    op->SetInput("GTBox", Input("GTBox"));
    op->SetInput("GTLabel", Input("GTLabel"));
    op->SetInput(framework::GradVarName("Loss"), OutputGrad("Loss"));

    op->SetAttrMap(Attrs());

    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetOutput(framework::GradVarName("GTBox"), {});
    op->SetOutput(framework::GradVarName("GTLabel"), {});
    return std::unique_ptr<framework::OpDesc>(op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(yolov3_loss, ops::Yolov3LossOp, ops::Yolov3LossOpMaker,
                  ops::Yolov3LossGradMaker);
REGISTER_OPERATOR(yolov3_loss_grad, ops::Yolov3LossOpGrad);
REGISTER_OP_CPU_KERNEL(yolov3_loss, ops::Yolov3LossKernel<float>,
                       ops::Yolov3LossKernel<double>);
REGISTER_OP_CPU_KERNEL(yolov3_loss_grad, ops::Yolov3LossGradKernel<float>,
                       ops::Yolov3LossGradKernel<double>);
