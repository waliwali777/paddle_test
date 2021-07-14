#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.tests.unittests.op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.framework import convert_np_dtype_to_dtype_

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUReduceProd(OpTest):
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {'dim': [0]}
        self.outputs = {
            'Out': self.inputs['X'].prod(axis=tuple(self.attrs['dim']))
        }

        # print('type(X)', type(self.inputs['X']))
        print('X.shape: ', self.inputs['X'].shape)
        print('Out.shape: ', self.outputs['Out'].shape)

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

    # def test_check_grad(self):
    #     self.check_grad_with_place(self.place, ['X'], 'Out')

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def init_dtype(self):
        self.dtype = np.float32


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUReduceProd6D(TestNPUReduceProd):
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {
            'X': np.random.random((5, 6, 2, 3, 4, 2)).astype(self.dtype)
        }
        self.attrs = {'dim': [2, 3, 4]}
        self.outputs = {
            'Out': self.inputs['X'].prod(axis=tuple(self.attrs['dim']))
        }

        # print('type(X)', type(self.inputs['X']))
        print('X.shape: ', self.inputs['X'].shape)
        print('Out.shape: ', self.outputs['Out'].shape)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUReduceProd8D(TestNPUReduceProd):
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {
            'X': np.random.random((2, 5, 3, 2, 2, 3, 4, 2)).astype(self.dtype)
        }
        self.attrs = {'dim': [2, 3, 4]}
        self.outputs = {
            'Out': self.inputs['X'].prod(axis=tuple(self.attrs['dim']))
        }

        # print('type(X)', type(self.inputs['X']))
        print('X.shape: ', self.inputs['X'].shape)
        print('Out.shape: ', self.outputs['Out'].shape)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUReduceProd2(TestNPUReduceProd):
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {'X': np.random.random((3, 5)).astype(self.dtype)}
        self.attrs = {'dim': [-1]}
        self.outputs = {
            'Out': self.inputs['X'].prod(axis=tuple(self.attrs['dim']))
        }

        # print('type(X)', type(self.inputs['X']))
        print('X.shape: ', self.inputs['X'].shape)
        print('Out.shape: ', self.outputs['Out'].shape)
        print('X: ', self.inputs['X'])
        print('Out: ', self.outputs['Out'])


if __name__ == '__main__':
    unittest.main()
