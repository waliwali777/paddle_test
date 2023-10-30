#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle.base import core
from paddle.pir_uitls import test_with_pir_api


class TestLogspaceOpCommonCase(OpTest):
    def setUp(self):
        self.op_type = "logspace"
        self.python_api = paddle.logspace
        self.init_data()

    def init_data(self):
        dtype = 'float32'
        self.inputs = {
            'Start': np.array([0]).astype(dtype),
            'Stop': np.array([10]).astype(dtype),
            'Num': np.array([11]).astype('int32'),
            'Base': np.array([2]).astype(dtype),
        }
        self.attrs = {'dtype': int(paddle.float32)}
        self.outputs = {'Out': np.power(2, np.arange(0, 11)).astype(dtype)}

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestLogspaceFP16Op(TestLogspaceOpCommonCase):
    def init_data(self):
        self.dtype = np.float16
        self.inputs = {
            'Start': np.array([0]).astype(self.dtype),
            'Stop': np.array([10]).astype(self.dtype),
            'Num': np.array([11]).astype('int32'),
            'Base': np.array([2]).astype(self.dtype),
        }
        self.attrs = {'dtype': int(paddle.float16)}
        self.outputs = {'Out': np.power(2, np.arange(0, 11)).astype(self.dtype)}


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestLogspaceBF16Op(OpTest):
    def setUp(self):
        self.op_type = "logspace"
        self.python_api = paddle.logspace
        self.init_data()

    def init_data(self):
        self.dtype = np.uint16
        self.np_dtype = np.float32
        self.inputs = {
            'Start': np.array([0]).astype(self.np_dtype),
            'Stop': np.array([10]).astype(self.np_dtype),
            'Num': np.array([11]).astype('int32'),
            'Base': np.array([2]).astype(self.np_dtype),
        }
        self.attrs = {'dtype': int(paddle.bfloat16)}
        self.outputs = {
            'Out': np.power(2, np.arange(0, 11)).astype(self.np_dtype)
        }

        self.inputs["Start"] = convert_float_to_uint16(self.inputs["Start"])
        self.inputs["Stop"] = convert_float_to_uint16(self.inputs["Stop"])
        self.inputs["Base"] = convert_float_to_uint16(self.inputs["Base"])
        self.outputs["Out"] = convert_float_to_uint16(self.outputs["Out"])
        self.place = core.CUDAPlace(0)

    def test_check_output(self):
        self.check_output_with_place(self.place, check_pir=True)


class TestLogspaceOpReverseCase(TestLogspaceOpCommonCase):
    def init_data(self):
        dtype = 'float32'
        self.inputs = {
            'Start': np.array([10]).astype(dtype),
            'Stop': np.array([0]).astype(dtype),
            'Num': np.array([11]).astype('int32'),
            'Base': np.array([2]).astype(dtype),
        }
        self.attrs = {'dtype': int(paddle.float32)}
        self.outputs = {'Out': np.power(2, np.arange(10, -1, -1)).astype(dtype)}


class TestLogspaceOpNumOneCase(TestLogspaceOpCommonCase):
    def init_data(self):
        dtype = 'float32'
        self.inputs = {
            'Start': np.array([10]).astype(dtype),
            'Stop': np.array([0]).astype(dtype),
            'Num': np.array([1]).astype('int32'),
            'Base': np.array([2]).astype(dtype),
        }
        self.attrs = {'dtype': int(paddle.float32)}
        self.outputs = {'Out': np.power(2, np.array([10])).astype(dtype)}


class TestLogspaceOpMinusBaseCase(TestLogspaceOpCommonCase):
    def init_data(self):
        dtype = 'float32'
        self.inputs = {
            'Start': np.array([0]).astype(dtype),
            'Stop': np.array([10]).astype(dtype),
            'Num': np.array([11]).astype('int32'),
            'Base': np.array([-2]).astype(dtype),
        }
        self.attrs = {'dtype': int(paddle.float32)}
        self.outputs = {'Out': np.power(-2, np.arange(0, 11)).astype(dtype)}


class TestLogspaceOpZeroBaseCase(TestLogspaceOpCommonCase):
    def init_data(self):
        dtype = 'float32'
        self.inputs = {
            'Start': np.array([0]).astype(dtype),
            'Stop': np.array([10]).astype(dtype),
            'Num': np.array([11]).astype('int32'),
            'Base': np.array([0]).astype(dtype),
        }
        self.attrs = {'dtype': int(paddle.float32)}
        self.outputs = {'Out': np.power(0, np.arange(0, 11)).astype(dtype)}


class TestLogspaceAPI(unittest.TestCase):
    @test_with_pir_api
    def test_variable_input1(self):
        paddle.enable_static()
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            start = paddle.full(shape=[1], fill_value=0, dtype='float32')
            stop = paddle.full(shape=[1], fill_value=10, dtype='float32')
            num = paddle.full(shape=[1], fill_value=5, dtype='int32')
            base = paddle.full(shape=[1], fill_value=2, dtype='float32')
            out = paddle.logspace(start, stop, num, base, dtype='float32')

        exe = paddle.static.Executor()
        res = exe.run(prog, fetch_list=[out])
        np_res = np.logspace(0, 10, 5, base=2, dtype='float32')
        self.assertEqual((res == np_res).all(), True)
        paddle.disable_static()

    def test_variable_input2(self):
        paddle.disable_static()
        start = paddle.full(shape=[1], fill_value=0, dtype='float32')
        stop = paddle.full(shape=[1], fill_value=10, dtype='float32')
        num = paddle.full(shape=[1], fill_value=5, dtype='int32')
        base = paddle.full(shape=[1], fill_value=2, dtype='float32')
        out = paddle.logspace(start, stop, num, base, dtype='float32')
        np_res = np.logspace(0, 10, 5, base=2, dtype='float32')
        self.assertEqual((out.numpy() == np_res).all(), True)
        paddle.enable_static()

    @test_with_pir_api
    def test_dtype(self):
        paddle.enable_static()
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            out_1 = paddle.logspace(0, 10, 5, 2, dtype='float32')
            out_2 = paddle.logspace(0, 10, 5, 2, dtype=np.float32)

        exe = paddle.static.Executor()
        res_1, res_2 = exe.run(prog, fetch_list=[out_1, out_2])
        np.testing.assert_array_equal(res_1, res_2)
        paddle.disable_static()

    def test_name(self):
        with paddle.static.program_guard(paddle.static.Program()):
            out = paddle.logspace(
                0, 10, 5, 2, dtype='float32', name='logspace_res'
            )
            assert 'logspace_res' in out.name

    def test_imperative(self):
        paddle.disable_static()
        out1 = paddle.logspace(0, 10, 5, 2, dtype='float32')
        np_out1 = np.logspace(0, 10, 5, base=2, dtype='float32')
        out2 = paddle.logspace(0, 10, 5, 2, dtype='int32')
        np_out2 = np.logspace(0, 10, 5, base=2, dtype='int32')
        out3 = paddle.logspace(0, 10, 200, 2, dtype='int32')
        np_out3 = np.logspace(0, 10, 200, base=2, dtype='int32')
        paddle.enable_static()
        self.assertEqual((out1.numpy() == np_out1).all(), True)
        self.assertEqual((out2.numpy() == np_out2).all(), True)
        self.assertEqual((out3.numpy() == np_out3).all(), True)


class TestLogspaceOpError(unittest.TestCase):
    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):

            def test_dtype():
                paddle.logspace(0, 10, 1, 2, dtype="int8")

            self.assertRaises(TypeError, test_dtype)

            def test_dtype1():
                paddle.logspace(0, 10, 1.33, 2, dtype="int32")

            self.assertRaises(TypeError, test_dtype1)

            def test_start_type():
                paddle.logspace([0], 10, 1, 2, dtype="float32")

            self.assertRaises(TypeError, test_start_type)

            def test_end_type():
                paddle.logspace(0, [10], 1, 2, dtype="float32")

            self.assertRaises(TypeError, test_end_type)

            def test_num_type():
                paddle.logspace(0, 10, [0], 2, dtype="float32")

            self.assertRaises(TypeError, test_num_type)

            def test_start_dtype():
                start = paddle.static.data(
                    shape=[1], dtype="float64", name="start"
                )
                paddle.logspace(start, 10, 1, 2, dtype="float32")

            self.assertRaises(ValueError, test_start_dtype)

            def test_end_dtype():
                end = paddle.static.data(shape=[1], dtype="float64", name="end")
                paddle.logspace(0, end, 1, 2, dtype="float32")

            self.assertRaises(ValueError, test_end_dtype)

            def test_num_dtype():
                num = paddle.static.data(
                    shape=[1], dtype="float32", name="step"
                )
                paddle.logspace(0, 10, num, 2, dtype="float32")

            self.assertRaises(TypeError, test_num_dtype)

            def test_base_dtype():
                base = paddle.static.data(
                    shape=[1], dtype="float64", name="end"
                )
                paddle.logspace(0, 10, 1, base, dtype="float32")

            self.assertRaises(ValueError, test_base_dtype)


if __name__ == "__main__":
    unittest.main()
