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
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard
from op_test import OpTest


class TestBincountOpAPI(unittest.TestCase):
    """Test bincount api."""

    def test_static_graph(self):
        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            inputs = fluid.data(name='input', dtype='int64', shape=[5])
            output = paddle.bincount(inputs)
            place = fluid.CPUPlace()
            if fluid.core.is_compiled_with_cuda():
                place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(startup_program)
            img = np.array([0, 1, 1, 3, 2, 1, 7]).astype(np.int64)
            res = exe.run(train_program,
                          feed={'input': img},
                          fetch_list=[output])
            actual = np.array(res[0])
            expected = np.array([1, 3, 1, 1, 0, 0, 0, 1]).astype(np.int64)
            self.assertTrue(
                (actual == expected).all(),
                msg='bincount output is wrong, out =' + str(actual))

    def test_dygraph(self):
        with fluid.dygraph.guard():
            inputs_np = np.array([0, 1, 1, 3, 2, 1, 7]).astype(np.int64)
            inputs = fluid.dygraph.to_variable(inputs_np)
            actual = paddle.bincount(inputs)
            expected = np.array([1, 3, 1, 1, 0, 0, 0, 1]).astype(np.int64)
            self.assertTrue(
                (actual.numpy() == expected).all(),
                msg='bincount output is wrong, out =' + str(actual.numpy()))


class TestBincountOpError(unittest.TestCase):
    """Test bincount op error."""

    def run_network(self, net_func):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            net_func()
            exe = fluid.Executor()
            exe.run(main_program)

    def test_input_value_error(self):
        """Test input tensor should only contain non-negative ints."""

        def net_func():
            input_value = paddle.to_tensor([1, 2, 3, 4, -5])
            paddle.bincount(input_value)

        with self.assertRaises(ValueError):
            self.run_network(net_func)

    def test_input_shape_error(self):
        """Test input tensor should be 1-D tansor."""

        def net_func():
            input_value = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
            paddle.bincount(input_value)

        with self.assertRaises(ValueError):
            self.run_network(net_func)

    def test_minlength_value_error(self):
        """Test minlength is non-negative ints."""

        def net_func():
            input_value = paddle.to_tensor([1, 2, 3, 4, 5])
            paddle.bincount(input_value, minlength=-1)

        with self.assertRaises(IndexError):
            self.run_network(net_func)

    def test_input_type_errors(self):
        """Test input tensor should only contain non-negative ints."""

        def net_func():
            input_value = paddle.to_tensor([1., 2., 3., 4., 5.])
            paddle.bincount(input_value)

        with self.assertRaises(ValueError):
            self.run_network(net_func)
    
    def test_weights_shape_error(self):
        """Test weights tensor should have the same shape as input tensor."""

        def net_func():
            input_value = paddle.to_tensor([1, 2, 3, 4, 5])
            weights = paddle.to_tensor([1, 1, 1, 1, 1, 1])
            paddle.bincount(input_value, weights=weights)

        with self.assertRaises(ValueError):
            self.run_network(net_func)


class TestBincountOp(OpTest):
    def setUp(self):
        self.op_type = "bincount"
        self.init_test_case()
        np_input = np.random.randint(low=0, high=20, size=self.in_shape)
        np_weights = np.random.randint(low=0, high=20, size=self.w_shape)
        self.inputs = {"X": np_input, "Weights": np_weights}
        self.init_attrs()
        Out = np.bincount(
            np_input, weights=np_weights, minlength=self.minlength)
        self.outputs = {"Out": Out}

    def init_test_case(self):
        self.in_shape = 10
        self.w_shape = 10
        self.minlength = 0


    def init_attrs(self):
        self.attrs = {"minlength": self.minlength}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
