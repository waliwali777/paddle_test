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

import unittest
from functools import partial
from typing import List

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertPad3d(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1():
            return np.ones([1, 1, 3, 64, 64]).astype(np.float32)

        def generate_paddings(p):
            return np.array(p).astype(np.int32)

        for value in [True, False]:
            for paddings in [
                [0, 0, 0, 0, 1, 1],
                [0, 0, 1, 2, 3, 4],
                [1, 1, 1, 1, 1, 1],
                [0, 0, -1, -1, 1, 1],
            ]:
                for mode in ['tensor', 'list']:
                    if mode == 'list':
                        dics = [{"value": value, "paddings": paddings, "data_format": "NCDHW"}, {}]
                        ops_config = [
                            {
                                "op_type": "pad3d",
                                "op_inputs": {"X": ["input_data"]},
                                "op_outputs": {"Out": ["output_data"]},
                                "op_attrs": dics[0],
                            }
                        ]
                    else:
                        dics = [{"value": value, "data_format": "NCDHW", "mode": "constant", "paddings": []}, {}]
                        ops_config = [
                            {
                                "op_type": "pad3d",
                                "op_inputs": {"X": ["input_data"],
                                              "Paddings": ["input_padding"]},
                                "op_outputs": {"Out": ["output_data"]},
                                "op_attrs": dics[0],
                            }
                        ]

                    ops = self.generate_op_config(ops_config)
                    inputs = {"input_data": TensorConfig(data_gen=partial(generate_input1))}
                    if mode == 'tensor':
                        inputs["input_padding"] = TensorConfig(data_gen=partial(generate_paddings, paddings))
                    for i in range(10):
                        program_config = ProgramConfig(
                            ops=ops,
                            weights={},
                            inputs=inputs,
                            outputs=["output_data"],
                        )

                        yield program_config

    def sample_predictor_configs(
            self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data": [1, 1, 3, 64, 64],
                "input_padding": [6],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [1, 1, 3, 64, 64],
                "input_padding": [6],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [1, 1, 3, 64, 64],
                "input_padding": [6],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (1, 3), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 3), 1e-3

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (1, 3), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 3), 1e-3

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
