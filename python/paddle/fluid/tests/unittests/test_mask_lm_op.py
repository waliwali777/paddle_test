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
from op_test import OpTest


class TestMaskLMOp(OpTest):
    def setUp(self):
        self.op_type = "mask_lm"

        self.init_test_case()

        self.inputs = {'X': (self.input_data, self.lod)}
        self.attrs = {
            'voc_size': self.voc_size,
            'mask_id': self.mask_id,
            'masked_prob': self.masked_prob,
            'fix_seed': self.fix_seed,
            'seed': self.seed
        }
        self.outputs = {
            'Out': (self.output, self.lod),
            'MaskPos': self.mask_pos,
            'Mask': self.mask_out
        }

    def test_check_output(self):
        if core.is_compiled_with_cuda() and core.op_support_gpu("mask_lm"):
            self.check_output_with_place(core.CUDAPlace(0), atol=1e-3)

    def init_test_case(self):
        self.mask_id = 0
        self.voc_size = 100000
        self.masked_prob = 0.0
        self.fix_seed = True
        self.seed = 0
        self.input_data = np.random.randint(
            1, self.voc_size, size=(30, 1)).astype("int32")

        self.lod = [[9, 4, 11, 6]]

        self.mask_pos = np.array([16]).reshape(-1, 1)
        self.mask_out = self.input_data[self.mask_pos[0][0]].reshape(-1, 1)

        self.output = self.input_data
        self.output[self.mask_pos[0][0]] = self.mask_id


class TestMaskLMOp1(TestMaskLMOp):
    def init_test_case(self):
        self.mask_id = 0
        self.voc_size = 100000
        self.masked_prob = 1.0
        self.fix_seed = True
        self.seed = 1
        self.input_data = np.array([
            41804, 33577, 91710, 56352, 47466, 27739, 97833, 21922, 91958,
            15117, 19348, 74047, 25609, 46964, 72075, 69982, 84402, 33889,
            36279, 37262, 51326, 81762, 80231, 29753, 37573, 58972, 94943,
            29062, 99629, 51213
        ]).astype("int32").reshape(-1, 1)

        self.lod = [[9, 4, 11, 6]]

        self.mask_out = self.input_data
        self.mask_pos = np.arange(30).reshape(-1, 1)

        self.output = np.array([
            41804, 33577, 0, 0, 0, 89689, 0, 0, 0, 0, 19348, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 64712, 0, 0, 0
        ]).astype("int32").reshape(-1, 1)


class TestMaskLMOp2(TestMaskLMOp):
    """
    def setUp(self):
        self.op_type = "mask_lm"

        self.init_test_case()

        self.inputs = {'X': (self.input_data, self.lod)}
        self.attrs = {
                'voc_size': self.voc_size,
                'mask_id': self.mask_id, 'masked_prob': 0.3,
                'seed': 1, 'fix_seed': True
                }
        self.outputs = {
            'Out': (self.output, self.lod),
            'Mask': (self.mask_data, self.lod)
        }

    def test_check_output(self):
        if core.is_compiled_with_cuda() and core.op_support_gpu("mask_lm"):
            self.check_output_with_place(core.CUDAPlace(0), atol=1e-3)
    """

    def init_test_case(self):
        self.mask_id = 0
        self.seed = 1
        self.masked_prob = 0.3
        self.fix_seed = True
        self.voc_size = 100000
        self.input_data = np.array([
            41804, 33577, 91710, 56352, 47466, 27739, 97833, 21922, 91958,
            15117, 19348, 74047, 25609, 46964, 72075, 69982, 84402, 33889,
            36279, 37262, 51326, 81762, 80231, 29753, 37573, 58972, 94943,
            29062, 99629, 51213
        ]).astype("int32").reshape(-1, 1)
        self.lod = [[9, 4, 11, 6]]

        self.output = np.array([
            41804, 0, 91710, 56352, 47466, 0, 97833, 21922, 0, 15117, 0, 74047,
            0, 46964, 72075, 69982, 84402, 33889, 36279, 37262, 0, 81762, 80231,
            29753, 37573, 58972, 0, 29062, 99629, 0
        ]).astype("int32").reshape(-1, 1)

        self.mask_out = np.array(
            [41804, 33577, 27739, 91958, 19348, 25609, 51326, 94943,
             51213]).astype("int32").reshape(-1, 1)

        self.mask_pos = np.array([0, 1, 5, 8, 10, 12, 20, 26,
                                  29]).astype("int64").reshape(-1, 1)


if __name__ == '__main__':
    unittest.main()
