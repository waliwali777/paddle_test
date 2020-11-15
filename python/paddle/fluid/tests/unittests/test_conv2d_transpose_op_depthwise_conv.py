#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import numpy as np

import paddle.fluid.core as core
import paddle.fluid as fluid
from op_test import OpTest
from test_conv2d_transpose_op import TestConv2dTransposeOp


class TestDepthwiseConvTranspose(TestConv2dTransposeOp):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.dilations = [1, 1]
        self.input_size = [1, 8, 16, 16]  # NCHW
        self.groups = 8
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [self.input_size[1], f_c, 4, 4]
        self.op_type = "depthwise_conv2d_transpose"


class TestDepthwiseConvTransposeAsymmetricPad(TestConv2dTransposeOp):
    def init_test_case(self):
        self.pad = [1, 0, 1, 2]
        self.stride = [2, 2]
        self.dilations = [1, 1]
        self.input_size = [1, 8, 16, 16]  # NCHW
        self.groups = 8
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [self.input_size[1], f_c, 3, 3]
        self.op_type = "depthwise_conv2d_transpose"
        self.data_format = 'NCHW'


class TestDepthwiseConvTransposeSAMEPad(TestConv2dTransposeOp):
    def init_test_case(self):
        self.stride = [2, 2]
        self.dilations = [1, 1]
        self.input_size = [1, 8, 16, 16]  # NHWC
        self.groups = 8
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [self.input_size[1], f_c, 3, 3]
        self.op_type = "depthwise_conv2d_transpose"
        self.padding_algorithm = 'SAME'


class TestDepthwiseConvTransposeVALIDPad(TestConv2dTransposeOp):
    def init_test_case(self):
        self.stride = [2, 2]
        self.dilations = [1, 1]
        self.input_size = [1, 8, 16, 16]  # NHWC
        self.groups = 8
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [self.input_size[1], f_c, 3, 3]
        self.op_type = "depthwise_conv2d_transpose"
        self.padding_algorithm = 'VALID'


class TestDepthwiseConvTranspose_NHWC_4x4kernel(TestConv2dTransposeOp):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.dilations = [1, 1]
        self.input_size = [1, 16, 16, 8]  # NHWC
        self.groups = 8
        assert np.mod(self.input_size[3], self.groups) == 0
        f_c = self.input_size[3] // self.groups
        self.filter_size = [self.input_size[3], f_c, 4, 4]
        self.op_type = "depthwise_conv2d_transpose"
        self.data_format = 'NHWC'


class TestDepthwiseConvTranspose_NHWC_3x3kernel(TestConv2dTransposeOp):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.dilations = [1, 1]
        self.input_size = [2, 16, 16, 8]  # NHWC
        self.groups = 8
        assert np.mod(self.input_size[3], self.groups) == 0
        f_c = self.input_size[3] // self.groups
        self.filter_size = [self.input_size[3], f_c, 3, 3]
        self.op_type = "depthwise_conv2d_transpose"
        self.data_format = 'NHWC'


class TestDepthwiseConvTransposeAsymmetricPad_NHWC(TestConv2dTransposeOp):
    def init_test_case(self):
        self.pad = [1, 0, 1, 2]
        self.stride = [2, 2]
        self.dilations = [1, 1]
        self.input_size = [2, 16, 16, 8]  # NHWC
        self.groups = 8
        assert np.mod(self.input_size[3], self.groups) == 0
        f_c = self.input_size[3] // self.groups
        self.filter_size = [self.input_size[3], f_c, 3, 3]
        self.op_type = "depthwise_conv2d_transpose"
        self.data_format = 'NHWC'


if __name__ == '__main__':
    unittest.main()
