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

from auto_scan_test import PassAutoScanTest, SkipReasons
from program_config import OpConfig, TensorConfig, ProgramConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st


class TestFCMishMkldnnFusePass(PassAutoScanTest):
    def sample_program_config(self, draw):
        x_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=8), min_size=2, max_size=5))
        x_shape = [2, 1]
        x_rank = len(x_shape)
        in_num_col_dims = draw(st.integers(min_value=1, max_value=x_rank - 1))
        w_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=8), min_size=2, max_size=2))
        w_shape[0] = int(np.prod(x_shape[in_num_col_dims:]))
        w_shape = [1, 2]
        fc_bias_shape = [w_shape[1], ]
        if draw(st.booleans()):
            fc_bias_shape.insert(0, 1)
        fc_bias_shape = [2, ]

        ops_config = [{
            "op_type": "fc",
            "op_inputs": {
                "Input": ["fc_x"],
                "W": ["fc_w"],
                "Bias": ["fc_bias"]
            },
            "op_outputs": {
                "Out": ["fc_out"]
            },
            "op_attrs": {
                "activation_type": "",
                "padding_weights": False,
                "in_num_col_dims": in_num_col_dims,
                "use_mkldnn": True
            }
        }, {
            "op_type": "mish",
            "op_inputs": {
                "X": ["fc_out"]
            },
            "op_outputs": {
                "Out": ["mish_output"]
            },
            "op_attrs": {
            },
        }]

        ops = self.generate_op_config(ops_config)

        program_config = ProgramConfig(
            ops=ops,
            weights={
                "fc_w": TensorConfig(shape=w_shape),
                "fc_bias": TensorConfig(shape=fc_bias_shape),
            },
            inputs={"fc_x": TensorConfig(shape=x_shape), },
            outputs=["mish_output"])
        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_mkldnn=True)
        yield config, ["fc"], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(quant=False, passes=["fc_act_mkldnn_fuse_pass"])


if __name__ == "__main__":
    unittest.main()
