# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest
import numpy as np
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
from op_test import OpTest, _set_use_system_allocator
from paddle.fluid.framework import grad_var_name
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
import paddle


class TestBatchNorm(unittest.TestCase):
    def test_name(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu("batch_norm"):
            places.append(fluid.CUDAPlace(0))
        for p in places:
            with fluid.dygraph.guard(p):
                batch_norm1d = paddle.nn.BatchNorm1d(1, name="test")

    def test_error(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu("batch_norm"):
            places.append(fluid.CUDAPlace(0))
        for p in places:
            #paddle.disable_static()
            x_data_4 = np.random.random(size=(2, 1, 3, 3)).astype('float32')
            x_data_3 = np.random.random(size=(2, 1, 3)).astype('float32')

            def error1d():
                x_data_4 = np.random.random(size=(2, 1, 3, 3)).astype('float32')
                batch_norm1d = paddle.nn.BatchNorm1d(1)
                batch_norm1d(fluid.dygraph.to_variable(x_data_4))

            def error2d():
                x_data_3 = np.random.random(size=(2, 1, 3)).astype('float32')
                batch_norm2d = paddle.nn.BatchNorm2d(1)
                batch_norm2d(fluid.dygraph.to_variable(x_data_3))

            def error3d():
                x_data_4 = np.random.random(size=(2, 1, 3, 3)).astype('float32')
                batch_norm3d = paddle.nn.BatchNorm3d(1)
                batch_norm3d(fluid.dygraph.to_variable(x_data_4))

            with fluid.dygraph.guard(p):
                self.assertRaises(ValueError, error1d)
                self.assertRaises(ValueError, error2d)
                self.assertRaises(ValueError, error3d)

    def test_dygraph(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu("batch_norm"):
            places.append(fluid.CUDAPlace(0))
        for p in places:
            shape = [4, 10, 4, 4]

            def compute_v1(x, is_test, trainable_statistics):
                with fluid.dygraph.guard(p):
                    bn = fluid.dygraph.BatchNorm(
                        shape[1],
                        is_test=is_test,
                        trainable_statistics=trainable_statistics)
                    y = bn(fluid.dygraph.to_variable(x))
                return y.numpy()

            def compute_v2(x):
                with fluid.dygraph.guard(p):
                    bn = paddle.nn.BatchNorm2d(shape[1])
                    y = bn(fluid.dygraph.to_variable(x))
                return y.numpy()

            def compute_v3(x, is_test, trainable_statistics):
                with fluid.dygraph.guard(p):
                    bn = fluid.dygraph.BatchNorm(
                        shape[1],
                        is_test=is_test,
                        param_attr=fluid.ParamAttr(
                            initializer=fluid.initializer.Constant(1.0),
                            trainable=False),
                        bias_attr=fluid.ParamAttr(
                            initializer=fluid.initializer.Constant(0.0),
                            trainable=False),
                        trainable_statistics=trainable_statistics)
                    y = bn(fluid.dygraph.to_variable(x))
                return y.numpy()

            def compute_v4(x):
                with fluid.dygraph.guard(p):
                    bn = paddle.nn.BatchNorm2d(
                        shape[1], weight_attr=False, bias_attr=False)
                    y = bn(fluid.dygraph.to_variable(x))
                return y.numpy()

            x = np.random.randn(*shape).astype("float32")
            y1 = compute_v1(x, False, False)
            y2 = compute_v2(x)
            y3 = compute_v3(x, False, False)
            y4 = compute_v4(x)
            self.assertTrue(np.allclose(y1, y2))
            self.assertTrue(np.allclose(y3, y4))

    def test_static(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu("batch_norm"):
            places.append(fluid.CUDAPlace(0))
        for p in places:
            exe = fluid.Executor(p)
            shape = [4, 10, 16, 16]

            def compute_v1(x_np, is_test, trainable_statistics):
                with program_guard(Program(), Program()):
                    bn = fluid.dygraph.BatchNorm(
                        shape[1],
                        is_test=is_test,
                        trainable_statistics=trainable_statistics)
                    x = fluid.data(name='x', shape=x_np.shape, dtype=x_np.dtype)
                    y = bn(x)
                    exe.run(fluid.default_startup_program())
                    r = exe.run(feed={'x': x_np}, fetch_list=[y])[0]
                return r

            def compute_v2(x_np):
                with program_guard(Program(), Program()):
                    bn = paddle.nn.BatchNorm2d(shape[1])
                    x = fluid.data(name='x', shape=x_np.shape, dtype=x_np.dtype)
                    y = bn(x)
                    exe.run(fluid.default_startup_program())
                    r = exe.run(feed={'x': x_np}, fetch_list=[y])[0]
                return r

            x = np.random.randn(*shape).astype("float32")
            y1 = compute_v1(x, False, False)
            y2 = compute_v2(x)
            self.assertTrue(np.allclose(y1, y2))


if __name__ == '__main__':
    unittest.main()
