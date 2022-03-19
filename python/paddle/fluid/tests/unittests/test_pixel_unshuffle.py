# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from op_test import OpTest
import paddle
import paddle.nn.functional as F
import paddle.fluid.core as core
import paddle.fluid as fluid

paddle.enable_static()

def pixel_unshuffle_np(x, down_factor, data_format="NCHW"):
    if data_format == "NCHW":
        n, c, h, w = x.shape
        new_shape = (n, c, h // down_factor, down_factor, 
                           w // down_factor, down_factor)
        npresult = np.reshape(x, new_shape)
        npresult = npresult.transpose(0, 1, 3, 5, 2, 4)
        oshape = [n, c * down_factor * down_factor, h // down_factor, 
                                                    w // down_factor]
        npresult = np.reshape(npresult, oshape)
        return npresult
    else:
        n, h, w, c = x.shape
        new_shape = (n, h // down_factor, down_factor, 
                        w // down_factor, down_factor, c)
        npresult = np.reshape(x, new_shape)
        npresult = npresult.transpose(0, 1, 3, 5, 2, 4)
        oshape = [n, h // down_factor, 
                     w // down_factor, c * down_factor * down_factor]
        npresult = np.reshape(npresult, oshape)
        return npresult


class TestPixelUnshuffleOp(OpTest):
    def setUp(self):
        self.op_type = "pixel_unshuffle"
        self.init_data_format()
        n, c, h, w = 2, 1, 12, 12

        if self.format == "NCHW":
            shape = [n, c, h, w]
        if self.format == "NHWC":
            shape = [n, h, w, c]

        down_factor = 3

        x = np.random.random(shape).astype("float64")
        npresult = pixel_unshuffle_np(x, down_factor, self.format)

        self.inputs = {"X": x}
        self.outputs = {"Out": npresult}
        self.attrs = {"downscale_factor": down_factor, 
                      "data_format": self.format}

    def init_data_format(self):
        self.format = "NCHW"

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestChannelLast(TestPixelUnshuffleOp):
    def init_data_format(self):
        self.format = "NHWC"


class TestPixelUnshuffleAPI(unittest.TestCase):
    def setUp(self):
        self.x_1_np = np.random.random([2, 1, 12, 12]).astype("float64")
        self.x_2_np = np.random.random([2, 12, 12, 1]).astype("float64")
        self.out_1_np = pixel_unshuffle_np(self.x_1_np, 3)
        self.out_2_np = pixel_unshuffle_np(self.x_2_np, 3, "NHWC")

    def test_static_graph_functional(self):
        for use_cuda in ([False, True]
                         if core.is_compiled_with_cuda() else [False]):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()

            paddle.enable_static()
            x_1 = paddle.fluid.data(
                name="x", shape=[2, 1, 12, 12], dtype="float64")
            x_2 = paddle.fluid.data(
                name="x2", shape=[2, 12, 12, 1], dtype="float64")
            out_1 = F.pixel_unshuffle(x_1, 3)
            out_2 = F.pixel_unshuffle(x_2, 3, "NHWC")

            exe = paddle.static.Executor(place=place)
            res_1 = exe.run(fluid.default_main_program(),
                            feed={"x": self.x_1_np},
                            fetch_list=out_1,
                            use_prune=True)

            res_2 = exe.run(fluid.default_main_program(),
                            feed={"x2": self.x_2_np},
                            fetch_list=out_2,
                            use_prune=True)

            assert np.allclose(res_1, self.out_1_np)
            assert np.allclose(res_2, self.out_2_np)

    # same test between layer and functional in this op.
    def test_static_graph_layer(self):
        for use_cuda in ([False, True]
                         if core.is_compiled_with_cuda() else [False]):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()

            paddle.enable_static()
            x_1 = paddle.fluid.data(
                name="x", shape=[2, 1, 12, 12], dtype="float64")
            x_2 = paddle.fluid.data(
                name="x2", shape=[2, 12, 12, 1], dtype="float64")
            # init instance
            ps_1 = paddle.nn.PixelUnshuffle(3)
            ps_2 = paddle.nn.PixelUnshuffle(3, "NHWC")
            out_1 = ps_1(x_1)
            out_2 = ps_2(x_2)
            out_1_np = pixel_unshuffle_np(self.x_1_np, 3)
            out_2_np = pixel_unshuffle_np(self.x_2_np, 3, "NHWC")

            exe = paddle.static.Executor(place=place)
            res_1 = exe.run(fluid.default_main_program(),
                            feed={"x": self.x_1_np},
                            fetch_list=out_1,
                            use_prune=True)

            res_2 = exe.run(fluid.default_main_program(),
                            feed={"x2": self.x_2_np},
                            fetch_list=out_2,
                            use_prune=True)

            assert np.allclose(res_1, out_1_np)
            assert np.allclose(res_2, out_2_np)

    def run_dygraph(self, down_factor, data_format):

        n, c, h, w = 2, 1, 12, 12

        if data_format == "NCHW":
            shape = [n, c, h, w]
        if data_format == "NHWC":
            shape = [n, h, w, c]

        x = np.random.random(shape).astype("float64")

        npresult = pixel_unshuffle_np(x, down_factor, data_format)

        for use_cuda in ([False, True]
                         if core.is_compiled_with_cuda() else [False]):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()

            paddle.disable_static(place=place)

            pixel_unshuffle = paddle.nn.PixelUnshuffle(
                down_factor, data_format=data_format)
            result = pixel_unshuffle(paddle.to_tensor(x))

            self.assertTrue(np.allclose(result.numpy(), npresult))

            result_functional = F.pixel_unshuffle(
                paddle.to_tensor(x), 3, data_format)
            self.assertTrue(np.allclose(result_functional.numpy(), npresult))

    def test_dygraph1(self):
        self.run_dygraph(3, "NCHW")

    def test_dygraph2(self):
        self.run_dygraph(3, "NHWC")


class TestPixelUnshuffleError(unittest.TestCase):
    def test_error_functional(self):
        def error_downscale_factor():
            with paddle.fluid.dygraph.guard():
                x = np.random.random([2, 1, 12, 12]).astype("float64")
                pixel_unshuffle = F.pixel_unshuffle(paddle.to_tensor(x), 3.33)

        self.assertRaises(TypeError, error_downscale_factor)

        def error_data_format():
            with paddle.fluid.dygraph.guard():
                x = np.random.random([2, 1, 12, 12]).astype("float64")
                pixel_unshuffle = F.pixel_unshuffle(paddle.to_tensor(x), 3, "WOW")

        self.assertRaises(ValueError, error_data_format)

    def test_error_layer(self):
        def error_downscale_factor_layer():
            with paddle.fluid.dygraph.guard():
                x = np.random.random([2, 1, 12, 12]).astype("float64")
                ps = paddle.nn.PixelUnshuffle(3.33)

        self.assertRaises(TypeError, error_downscale_factor_layer)

        def error_data_format_layer():
            with paddle.fluid.dygraph.guard():
                x = np.random.random([2, 1, 12, 12]).astype("float64")
                ps = paddle.nn.PixelUnshuffle(3, "MEOW")

        self.assertRaises(ValueError, error_data_format_layer)


if __name__ == "__main__":
    unittest.main()
