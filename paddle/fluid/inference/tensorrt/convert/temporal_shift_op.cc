/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle {
namespace framework {
class Scope;

namespace proto {
class OpDesc;
}  // namespace proto
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace tensorrt {

/*
 * TemporalShiftOp.
 */
class TemporalShiftOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
#if IS_TRT_VERSION_GE(8200)
    VLOG(3) << "convert a fluid transpose op to tensorrt tranpose layer";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);

    const float shift_ratio =
        PADDLE_GET_CONST(float, op_desc.GetAttr("shift_ratio"));
    const int T = PADDLE_GET_CONST(int, op_desc.GetAttr("seg_num"));

    std::string data_format = "NCHW";
    if (op_desc.HasAttr("data_format")) {
      data_format =
          PADDLE_GET_CONST(std::string, op_desc.GetAttr("data_format"));
    }

    if (data_format == "NHWC") {
      // tanspose input to [N,C,H,W]
      auto transpose_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
      nvinfer1::Permutation perm{0, 3, 1, 2};
      transpose_layer->setFirstTranspose(perm);
      input = transpose_layer->getOutput(0);
    }

    auto input_dims = input->getDimensions();

    const int NT = input_dims.d[0];
    const int C = input_dims.d[1];
    const int H = input_dims.d[2];
    const int W = input_dims.d[3];
    const int N = NT / T;

    // Reshape input to [N,T,C,H,W]
    auto reshape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
    nvinfer1::Dims reshape_dims{5, { N, T, C, H, W }};
    reshape_layer->setReshapeDimensions(reshape_dims);

    // Pad input to [N,T+2,C,H,W]
    std::vector<int> pre_pad_v{0, 1, 0, 0, 0};
    std::vector<int> post_pad_v{0, 1, 0, 0, 0};
    nvinfer1::ITensor* pre_pad = Add1DConstantLayer(pre_pad_v);
    nvinfer1::ITensor* post_pad = Add1DConstantLayer(post_pad_v);

    int dims = 5;
    std::vector<int> zeros_v(dims, 0);
    auto const zeros = Add1DConstantLayer(zeros_v);

    nvinfer1::ITensor* start{};
    nvinfer1::ITensor* size{};

    start = TRT_ENGINE_ADD_LAYER(engine_,
                                 ElementWise,
                                 *zeros,
                                 *pre_pad,
                                 nvinfer1::ElementWiseOperation::kSUB)
                ->getOutput(0);

    auto const total_padding =
        TRT_ENGINE_ADD_LAYER(engine_,
                             ElementWise,
                             *pre_pad,
                             *post_pad,
                             nvinfer1::ElementWiseOperation::kSUM)
            ->getOutput(0);

    std::vector<int> input_shape_v(dims, 0);
    for (int i = 0; i < dims; i++) {
      input_shape_v[i] = input->getDimensions().d[i];
    }
    auto const input_shape = Add1DConstantLayer(input_shape_v);

    size = TRT_ENGINE_ADD_LAYER(engine_,
                                ElementWise,
                                *input_shape,
                                *total_padding,
                                nvinfer1::ElementWiseOperation::kSUM)
               ->getOutput(0);
    nvinfer1::Dims stride;
    stride.nbDims = dims;
    std::fill_n(stride.d, dims, 1);
    auto const& dummy = stride;
    auto* slice_layer =
        TRT_ENGINE_ADD_LAYER(engine_,
                             Slice,
                             *const_cast<nvinfer1::ITensor*>(input),
                             dummy,
                             dummy,
                             stride);
    slice_layer->setInput(1, *start);
    slice_layer->setInput(2, *size);
    slice_layer->setMode(nvinfer1::SliceMode::kFILL);

    // Slice Padded Tensor
    int slice_c = int(C * shift_ratio);
    int slice_c2 = int(C * shift_ratio * 2);
    auto slice_start1 = nvinfer1::Dims{5, { 0, 0, 0, 0, 0 }};
    auto slice_start2 = nvinfer1::Dims{5, { 0, 2, slice_c, 0, 0 }};
    auto slice_start3 = nvinfer1::Dims{5, { 0, 1, slice_c2, 0, 0 }};
    auto slice_size = nvinfer1::Dims{5, { N, T, slice_c, H, W }};
    auto slice_size2 = nvinfer1::Dims{5, { N, T, C - slice_c2, H, W }};
    auto slice_stride = nvinfer1::Dims{5, { 1, 1, 1, 1, 1 }};

    auto* slice1_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Slice, *slice_layer->getOutput(0), slice_start1, slice_size, slice_stride);
    auto* slice2_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Slice, *slice_layer->getOutput(0), slice_start2, slice_size, slice_stride);
    auto* slice3_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Slice, *slice_layer->getOutput(0), slice_start3, slice_size2, slice_stride);

    // Concatenate slices along the third dimension (C)
    nvinfer1::IConcatenationLayer* concat_layer;
    if (!slice_c) {
      nvinfer1::ITensor* concat_inputs[2] = {slice2_layer->getOutput(0),
                                             slice3_layer->getOutput(0)};
      concat_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Concatenation, concat_inputs, 2);
      concat_layer->setAxis(2);
    } else {
      nvinfer1::ITensor* concat_inputs[3] = {slice1_layer->getOutput(0),
                                             slice2_layer->getOutput(0),
                                             slice3_layer->getOutput(0)};
      concat_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Concatenation, concat_inputs, 3);
      concat_layer->setAxis(2);
    }

    // Reshape output to [N*T,C,H,W]
    nvinfer1::Dims output_shape{4, { N * T, C, H, W }};
    auto* reshape_layer3 =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *concat_layer->getOutput(0));
    reshape_layer3->setReshapeDimensions(output_shape);

    // Set output
    auto output_name = op_desc.Output("Out")[0];

    if (data_format == "NHWC") {
      // Transpose output to [N*T,C,H,W] -> [N*T,H,W,C]
      auto transpose_layer2 =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *reshape_layer3->getOutput(0));
      nvinfer1::Permutation permute_order{0, 2, 3, 1};
      transpose_layer2->setFirstTranspose(permute_order);
      RreplenishLayerAndOutput(
          transpose_layer2, "temporal_shift", {output_name}, test_mode);
    } else {
      RreplenishLayerAndOutput(
          reshape_layer3, "temporal_shift", {output_name}, test_mode);
    }
#else
    VLOG(3) << "Temporal shift is not supported when TensorRT < 8.2";
#endif
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(temporal_shift, TemporalShiftOpConverter);
