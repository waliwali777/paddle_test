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

import numpy as np
from .... import core
from ....framework import IrGraph
from ....framework import IrNode
from ....framework import _get_paddle_place

__all__ = ['QuantInt8MkldnnPass']


class QuantInt8MkldnnPass(object):
    """
    Convert QuantizationFreezePass generated IrGraph to MKL-DNN supported INT8
    IrGraph. Following transformations did in this pass:
        1. Convert int8 range weights with float32 data type, which are generated by
           the QuantizationFreezePass, to float32 range weights with float32 data type
           by using the corresponding scales. This conversion is because MKL-DNN INT8
           conv2d kernel and mul kernel now only support float32 weights input, hence
           weights quantization will happen inside the conv2d and mul INT8 kernel.
        2. Create the new conv2d or mul op with the converted weights and link its output
           to fake_dequantize_abs_max op's output and set conv2d's attribute "force_fp32
           _output" as true
        3. Transform fake_quantize_xx op to quantize op
        4. Remove fake_dequantize_abs_max op
    """

    def __init__(self, _scope=None, _place=None):
        r"""
        Args:
            scope(fluid.Scope): scope is used to initialize the new parameters.
            place(fluid.CPUPlace|str): place is used to initialize the new parameters.
            When it is string, it can be only 'cpu'.


        Examples:
        .. code-block:: python
            # The original graph will be rewrite.
            import paddle.fluid as fluid
            from paddle.fluid.contrib.slim.quantization \
                import QuantInt8MkldnnPass
            from paddle.fluid.framework import IrGraph
            from paddle.fluid import core

            graph = IrGraph(core.Graph(fluid.Program().desc), for_test=False)
            place = fluid.CPUPlace()
            mkldnn_pass = QuantInt8MkldnnPass(fluid.global_scope(),
            place)
            mkldnn_pass.apply(graph)
        """

        self._scope = _scope
        self._place = _get_paddle_place(_place)

        self._quantize_type = [
            'fake_quantize_moving_average_abs_max',
            'fake_quantize_range_abs_max',
        ]
        self._dequantize_type = ['fake_dequantize_max_abs']
        self._quantize_dequantize_type = [
            'fake_quantize_dequantize_moving_average_abs_max'
        ]

        self._quantizable_ops = ['conv2d', 'depthwise_conv2d', 'mul']
        self._conv_ops = ['conv2d', 'depthwise_conv2d']
        self._pool_ops = ['pool2d']

        self._in_scale = {}
        self._max_range = {}
        self._new_output = {}
        self._s8_max = 127

    def apply(self, graph):
        """
        Quantize the graph for running MKL-DNN INT8 inference. According
        to activation quantization type, the graph will transform fake
        quantize ops to quantize ops and remove the fake dequantize ops.

        Args:
            graph(IrGraph): the applied graph.
        """

        assert isinstance(
            graph, IrGraph
        ), 'graph must be the instance of IrGraph.'
        ops = graph.all_op_nodes()

        persistable_vars = [p.name() for p in graph.all_persistable_nodes()]
        # Collect the _in_scales and _max_range to calculate the new scales for MKL-DNN
        # INT8 conv2d and mul
        for op_node in ops:
            if op_node.name() in self._dequantize_type:
                input_name = op_node.input("X")[0]
                scale_name = op_node.input("Scale")[0]
                self._in_scale[input_name] = self._load_param(
                    self._scope, scale_name
                )[0]
                self._max_range[input_name] = op_node.op().attr("max_range")
                self._new_output[input_name] = op_node.output("Out")[0]

            if op_node.name() in self._quantize_dequantize_type:
                inputs = op_node.op().input_names()
                attrs = op_node.op().attr_names()
                input_name = op_node.input("X")[0]
                scale_name = op_node.input("InScale")[0]
                self._in_scale[input_name] = self._load_param(
                    self._scope, scale_name
                )[0]
                #  self._max_range[input_name] = op_node.op().attr("max_range")
                self._new_output[input_name] = op_node.output("Out")[0]

        for op_node in ops:
            if op_node.name() in self._quantizable_ops:
                if op_node.name() in self._conv_ops:
                    self._transform_to_conv_mkldnn(graph, op_node)
                elif op_node.name() in self._pool_ops:
                    self._transform_to_pool_mkldnn(graph, op_node)
                else:
                    self._transform_to_mul_mkldnn(graph, op_node)
            elif op_node.name() in self._quantize_type:
                self._transform_to_quantize_mkldnn(graph, op_node)
            elif op_node.name() in self._dequantize_type:
                self._remove_fake_dequantize_op(graph, op_node)
            self._remove_unused_var_nodes(graph)
        return graph

    def _transform_to_pool_mkldnn(self, graph, op):
        output_name = op.output("Out")[0]
        input_name = op.input("X")[0]

    def _transform_to_conv_mkldnn(self, graph, op_node):
        weight_name = op_node.input("Filter")[0]
        output_name = op_node.output("Output")[0]
        # Convert int8 range weights to fp32 range weights
        weight = self._load_param(self._scope, weight_name)
        w_fp32 = np.divide(
            np.multiply(weight, self._s8_max), self._max_range[output_name]
        )
        w_fp32 = w_fp32.reshape(weight.shape)
        self._restore_var(weight_name, w_fp32)
        input_var_node = graph._find_node_by_name(
            op_node.inputs, op_node.input("Input")[0]
        )
        weight_var_node = graph._find_node_by_name(op_node.inputs, weight_name)

        # Set fake_dequantize_abs_max's output as new output of conv2d
        output_var_node = graph._find_node_by_name(
            graph.all_var_nodes(), self._new_output[output_name]
        )
        attrs = {
            name: op_node.op().attr(name) for name in op_node.op().attr_names()
        }

        conv_op_node = graph.create_op_node(
            op_type='conv2d',
            attrs=attrs,
            inputs={'Input': input_var_node, 'Filter': weight_var_node},
            outputs={'Output': output_var_node},
        )

        # Based on the Quant's scales to calculate the scales of MKL-DNN INT8 conv2d
        scale_in = self._s8_max / self._in_scale[output_name]
        scale_w = []
        scale_w = [self._max_range[output_name] / self._s8_max]

        conv_op_node.set_attr("Scale_weights", scale_w)
        conv_op_node.set_attr("Scale_in", scale_in)
        conv_op_node.set_attr("Scale_out", 1.0)
        conv_op_node.set_attr("use_mkldnn", 1)
        conv_op_node.set_attr("force_fp32_output", 1)
        graph.link_to(input_var_node, conv_op_node)
        graph.link_to(weight_var_node, conv_op_node)
        graph.link_to(conv_op_node, output_var_node)
        graph.safe_remove_nodes(op_node)

    def _transform_to_mul_mkldnn(self, graph, op_node):
        # For MKL-DNN INT8 mul, input Y should be the weights
        weight_name = op_node.input("Y")[0]
        output_name = op_node.output("Out")[0]
        # Convert int8 range weights to fp32 range weights
        weight = self._load_param(self._scope, weight_name)
        w_fp32 = np.divide(
            np.multiply(weight, self._s8_max), self._max_range[output_name]
        )
        w_fp32 = w_fp32.reshape(weight.shape)
        self._restore_var(weight_name, w_fp32)
        input_var_node = graph._find_node_by_name(
            op_node.inputs, op_node.input("X")[0]
        )
        weight_var_node = graph._find_node_by_name(op_node.inputs, weight_name)

        # Set fake_dequantize_abs_max's output as new output of mul
        output_var_node = graph._find_node_by_name(
            graph.all_var_nodes(), self._new_output[output_name]
        )
        attrs = {
            name: op_node.op().attr(name) for name in op_node.op().attr_names()
        }

        mul_op_node = graph.create_op_node(
            op_type='mul',
            attrs=attrs,
            inputs={'X': input_var_node, 'Y': weight_var_node},
            outputs={'Out': output_var_node},
        )

        # Based on the Quant's scales to calculate MKL-DNN INT8 mul's scales
        scale_in = self._s8_max / self._in_scale[output_name]
        scale_w = []
        scale_w = [self._max_range[output_name] / self._s8_max]

        mul_op_node.set_attr("scale_y", scale_w)
        mul_op_node.set_attr("scale_x", scale_in)
        mul_op_node.set_attr("scale_out", 1.0)
        mul_op_node.set_attr("use_mkldnn", 1)
        mul_op_node.set_attr("force_fp32_output", 1)
        graph.link_to(input_var_node, mul_op_node)
        graph.link_to(weight_var_node, mul_op_node)
        graph.link_to(mul_op_node, output_var_node)
        graph.safe_remove_nodes(op_node)

    def _transform_to_quantize_mkldnn(self, graph, op_node):
        """
        Transform fake_quantize_xx op to quantize mkldnn op in the graph.
        """
        input_var_node = graph._find_node_by_name(
            op_node.inputs, op_node.input("X")[0]
        )
        output_var_node = graph._find_node_by_name(
            op_node.outputs, op_node.output("Out")[0]
        )
        scale_in = (
            self._s8_max
            / self._load_param(self._scope, op_node.input("InScale")[0])[0]
        )
        quant_op_node = graph.create_op_node(
            op_type='quantize',
            attrs={
                'data_format': 'MKLDNNLAYOUT',
                'use_mkldnn': 1,
                'Scale': scale_in,
                'is_negative_input': 1,
            },
            inputs={'Input': input_var_node},
            outputs={'Output': output_var_node},
        )
        graph.link_to(input_var_node, quant_op_node)
        graph.link_to(quant_op_node, output_var_node)
        graph.safe_remove_nodes(op_node)

    def _remove_fake_dequantize_op(self, graph, op_node):
        input_var_node = graph._find_node_by_name(
            op_node.inputs, op_node.input("X")[0]
        )
        graph.safe_remove_nodes(op_node)

    def _load_param(self, scope, param_name):
        return np.array(scope.find_var(param_name).get_tensor())

    def _restore_var(self, name, array):
        tensor = self._scope.find_var(name).get_tensor()
        tensor.set(array, self._place)

    def _remove_unused_var_nodes(self, graph):
        all_used_vars = set()
        ops = graph.all_op_nodes()
        for op_node in ops:
            for input_node in op_node.inputs:
                all_used_vars.add(input_node)
            for output_node in op_node.outputs:
                all_used_vars.add(output_node)

        all_used_vars = {n.node for n in all_used_vars}
        all_unused_vars = {
            n
            for n in filter(
                lambda node: node.node not in all_used_vars,
                graph.all_var_nodes(),
            )
        }
        graph.safe_remove_nodes(all_unused_vars)
