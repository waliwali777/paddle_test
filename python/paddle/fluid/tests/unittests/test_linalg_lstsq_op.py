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
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core


class LinalgLstsqTestCase(unittest.TestCase):
    def setUp(self):
        self.devices = ["cpu"]
        if core.is_compiled_with_cuda():
            self.devices.append("gpu:0")
        self.init_config()
        self.generate_input()
        self.generate_output()

    def init_config(self):
        self.dtype = 'float64'
        self.rcond = 1e-15
        self.driver = "gels"
        self._input_shape_1 = (5, 4)
        self._input_shape_2 = (5, 3)

    def generate_input(self):
        self._input_data_1 = np.random.random(self._input_shape_1).astype(
            self.dtype)
        self._input_data_2 = np.random.random(self._input_shape_2).astype(
            self.dtype)

    def generate_output(self):
        if len(self._input_shape_1) == 2:
            out = np.linalg.lstsq(
                self._input_data_1, self._input_data_2, rcond=self.rcond)
        elif len(self._input_shape_1) == 3:
            out = np.linalg.lstsq(
                self._input_data_1[0], self._input_data_2[0], rcond=self.rcond)

        self._output_solution = out[0]
        self._output_residuals = out[1]
        self._output_rank = out[2]
        self._output_sg_values = out[3]

    def test_dygraph(self):
        paddle.disable_static()
        for dev in self.devices:
            paddle.set_device(dev)
            place = paddle.CPUPlace() if dev == "cpu" else paddle.CUDAPlace(0)
            x = paddle.to_tensor(
                self._input_data_1, place=place, dtype=self.dtype)
            y = paddle.to_tensor(
                self._input_data_2, place=place, dtype=self.dtype)
            results = paddle.linalg.lstsq(
                x, y, rcond=self.rcond, driver=self.driver)
            self.assert_np_close(results)

    def test_static(self):
        paddle.enable_static()
        for dev in self.devices:
            paddle.set_device(dev)
            place = fluid.CPUPlace() if dev == "cpu" else fluid.CUDAPlace(0)
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                x = paddle.fluid.data(
                    name="x",
                    shape=self._input_shape_1,
                    dtype=self._input_data_1.dtype)
                y = paddle.fluid.data(
                    name="y",
                    shape=self._input_shape_2,
                    dtype=self._input_data_2.dtype)
                results = paddle.linalg.lstsq(
                    x, y, rcond=self.rcond, driver=self.driver)
                exe = fluid.Executor(place)
                fetches = exe.run(
                    fluid.default_main_program(),
                    feed={"x": self._input_data_1,
                          "y": self._input_data_2},
                    fetch_list=[results])
                self.assert_np_close(fetches)

    def assert_np_close(self, results):
        np.testing.assert_allclose(results[0], self._output_solution, rtol=1e-3)
        if self._input_shape_1[-2] > self._input_shape_1[
                -1] and self._output_rank == self._input_shape_1[-1]:
            np.testing.assert_allclose(
                results[1], self._output_residuals, rtol=1e-5)

        if self.driver in ("gelsy", "gelsd", "gelss"):
            np.testing.assert_allclose(results[2], self._output_rank, rtol=1e-5)

        if self.driver in ("gelsd", "gelss"):
            np.testing.assert_allclose(
                results[3], self._output_sg_values, rtol=1e-5)


class LinalgLstsqTestCase(LinalgLstsqTestCase):
    def init_config(self):
        self.dtype = 'float64'
        self.rcond = 1e-15
        self.driver = "gels"
        self._input_shape_1 = (5, 10)
        self._input_shape_2 = (5, 5)


class LinalgLstsqTestCaseRcond(LinalgLstsqTestCase):
    def init_config(self):
        self.dtype = 'float64'
        self.rcond = 0.1
        self.driver = "gels"
        self._input_shape_1 = (3, 2)
        self._input_shape_2 = (3, 3)


class LinalgLstsqTestCaseGelsFloat32(LinalgLstsqTestCase):
    def init_config(self):
        self.dtype = 'float32'
        self.rcond = 1e-15
        self.driver = "gels"
        self._input_shape_1 = (10, 5)
        self._input_shape_2 = (10, 2)


class LinalgLstsqTestCaseGelssFloat64(LinalgLstsqTestCase):
    def init_config(self):
        self.dtype = 'float64'
        self.rcond = 1e-15
        self.driver = "gelss"
        self._input_shape_1 = (5, 5)
        self._input_shape_2 = (5, 1)
        self.devices = ["cpu"]


class LinalgLstsqTestCaseGelsyFloat32(LinalgLstsqTestCase):
    def init_config(self):
        self.dtype = 'float32'
        self.rcond = 1e-15
        self.driver = "gelsy"
        self._input_shape_1 = (8, 2)
        self._input_shape_2 = (8, 10)
        self.devices = ["cpu"]


class LinalgLstsqTestCaseBatch1(LinalgLstsqTestCase):
    def init_config(self):
        self.dtype = 'float32'
        self.rcond = 1e-15
        self.driver = None
        self._input_shape_1 = (2, 3, 10)
        self._input_shape_2 = (2, 3, 4)


class LinalgLstsqTestCaseBatch2(LinalgLstsqTestCase):
    def init_config(self):
        self.dtype = 'float64'
        self.rcond = 1e-15
        self.driver = "gelss"
        self._input_shape_1 = (2, 8, 6)
        self._input_shape_2 = (2, 8, 2)


class LinalgLstsqTestCaseLarge1(LinalgLstsqTestCase):
    def init_config(self):
        self.dtype = 'float64'
        self.rcond = 1e-15
        self.driver = "gelsd"
        self._input_shape_1 = (200, 100)
        self._input_shape_2 = (200, 50)
        self.devices = ["cpu"]


if __name__ == '__main__':
    unittest.main()
