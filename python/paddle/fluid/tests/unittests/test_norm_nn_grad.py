#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
import gradient_checker

from decorator_helper import prog_scope


class TestInstanceNormDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        prog = fluid.Program()
        with fluid.program_guard(prog):
            np.random.seed()
            shape = [2, 3, 4, 5]
            dtype = "float32"
            eps = 0.005
            atol = 1e-4
            x = layers.create_parameter(dtype=dtype, shape=shape, name='x')
            z = fluid.layers.instance_norm(input=x)
            x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
            gradient_checker.double_grad_check(
                [x], z, x_init=x_arr, atol=atol, place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestInstanceNormDoubleGradCheckWithoutParamBias(
        TestInstanceNormDoubleGradCheck):
    @prog_scope()
    def func(self, place):
        prog = fluid.Program()
        with fluid.program_guard(prog):
            np.random.seed()
            shape = [2, 3, 4, 5]
            dtype = "float32"
            eps = 0.005
            atol = 1e-4
            x = layers.create_parameter(dtype=dtype, shape=shape, name='x')
            z = fluid.layers.instance_norm(
                input=x, param_attr=False, bias_attr=False)
            x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
            gradient_checker.double_grad_check(
                [x], z, x_init=x_arr, atol=atol, place=place, eps=eps)


class TestBatchNormDoubleGradCheck(unittest.TestCase):
    def setUp(self):
        self.init_test()

    def init_test(self):
        self.data_layout = 'NCHW'
        self.use_global_stats = False

    @prog_scope()
    def func(self, place):
        prog = fluid.Program()
        with fluid.program_guard(prog):
            np.random.seed()
            shape = [2, 3, 4, 5]
            dtype = "float32"
            eps = 0.005
            atol = 1e-4
            x = layers.create_parameter(dtype=dtype, shape=shape, name='x')
            z = fluid.layers.batch_norm(
                input=x,
                data_layout=self.data_layout,
                use_global_stats=self.use_global_stats)
            x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
            gradient_checker.double_grad_check(
                [x], z, x_init=x_arr, atol=atol, place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestBatchNormDoubleGradCheckCase1(TestBatchNormDoubleGradCheck):
    def init_test(self):
        self.data_layout = 'NHWC'
        self.use_global_stats = False


class TestBatchNormDoubleGradCheckCase2(TestBatchNormDoubleGradCheck):
    def init_test(self):
        self.data_layout = 'NCHW'
        self.use_global_stats = True


class TestBatchNormDoubleGradCheckCase3(TestBatchNormDoubleGradCheck):
    def init_test(self):
        self.data_layout = 'NHWC'
        self.use_global_stats = True


if __name__ == "__main__":
    unittest.main()
