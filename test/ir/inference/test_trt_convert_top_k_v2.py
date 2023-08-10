# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from typing import Any, Dict, List

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertTopKV2Test(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        if len(inputs['input_data'].shape) <= attrs[0]['axis']:
            return False
        axis = attrs[0]['axis']
        axis = axis if axis >= 0 else axis + len(inputs['input_data'].shape)
        if inputs['input_data'].shape[axis] <= attrs[0]['k']:
            return False
        return True

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824

        def generate_input1(dims, batch, attrs: List[Dict[str, Any]]):
            if dims == 1:
                return np.random.random([3]).astype(np.float32)
            elif dims == 2:
                return np.random.random([3, 32]).astype(np.float32)
            elif dims == 3:
                return np.random.random([3, 32, 32]).astype(np.float32)
            else:
                return np.random.random([batch, 32, 32, 32]).astype(np.float32)

        for dims in [1, 2, 3, 4]:
            for batch in [1, 4]:
                for k in [1, 3]:
                    for axis in [-1, 1, 0, 2, 3]:
                        for largest in [True, False]:
                            for sort in [True, False]:
                                self.dims = dims
                                self.sort = sort
                                self.axis = axis
                                dics = [
                                    {
                                        "k": k,
                                        "axis": axis,
                                        "largest": largest,
                                        "sorted": sort,
                                    }
                                ]
                                ops_config = [
                                    {
                                        "op_type": "top_k_v2",
                                        "op_inputs": {"X": ["input_data"]},
                                        "op_outputs": {
                                            "Out": ["output_data"],
                                            "Indices": ["indices_data"],
                                        },
                                        "op_attrs": dics[0],
                                        "outputs_dtype": {
                                            "indices_data": np.int32
                                        },
                                    }
                                ]
                                ops = self.generate_op_config(ops_config)

                                program_config = ProgramConfig(
                                    ops=ops,
                                    weights={},
                                    inputs={
                                        "input_data": TensorConfig(
                                            data_gen=partial(
                                                generate_input1,
                                                dims,
                                                batch,
                                                dics,
                                            )
                                        )
                                    },
                                    outputs=["output_data", "indices_data"],
                                )

                                yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {"input_data": [1]}
                self.dynamic_shape.max_input_shape = {"input_data": [64]}
                self.dynamic_shape.opt_input_shape = {"input_data": [32]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 1]}
                self.dynamic_shape.max_input_shape = {"input_data": [4, 64]}
                self.dynamic_shape.opt_input_shape = {"input_data": [3, 10]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 1, 1]}
                self.dynamic_shape.max_input_shape = {"input_data": [4, 64, 64]}
                self.dynamic_shape.opt_input_shape = {"input_data": [3, 10, 10]}
            else:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 3, 16, 16]
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [4, 32, 32, 32]
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [4, 3, 32, 32]
                }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if not dynamic_shape and (self.dims == 1 or self.axis == 0):
                return 0, 4
            if not self.sort:
                return 0, 4
            return 1, 3

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-3

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-3

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
