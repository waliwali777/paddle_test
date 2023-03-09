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

import warnings

import paddle
from paddle import _C_ops

from ..fluid import core, framework, unique_name
from ..fluid.dygraph import no_grad
from ..fluid.layer_helper import LayerHelper
from ..framework import in_dygraph_mode
from .optimizer import Optimizer

__all__ = []


class Adadelta(Optimizer):
    r"""
    **Notes: This API does not support sparse parameter optimization.**

    Adadelta Optimizer. Please refer to this for details:
    `ADADELTA: AN ADAPTIVE LEARNING RATE METHOD <https://arxiv.org/abs/1212.5701>`_.

    The update is done as follows:

    .. math::

        E(g_t^2) &= \rho * E(g_{t-1}^2) + (1-\rho) * g^2

        learning\_rate &= \sqrt{ ( E(dx_{t-1}^2) + \epsilon ) / ( E(g_t^2) + \epsilon ) }

        E(dx_t^2) &= \rho * E(dx_{t-1}^2) + (1-\rho) * (-g*learning\_rate)^2

    Args:
        learning_rate (float|Tensor|LearningRateDecay, optional): The learning rate used to update ``Parameter``.
            It can be a float value, a ``Tensor`` with a float type or a LearningRateDecay. The default value is 0.001.
        epsilon (float): a small float number for numeric stability. Default 1.0e-6.
        rho (float): a floating point value indicating the decay rate. Default 0.95.
        parameters (list|tuple, optional): List/Tuple of ``Tensor`` to update to minimize ``loss``. \
            This parameter is required in dygraph mode. And you can specify different options for \
            different parameter groups such as the learning rate, weight decay, etc, \
            then the parameters are list of dict. Note that the learning_rate in paramter groups \
            represents the scale of base learning_rate. \
            The default value is None in static graph mode, at this time all parameters will be updated.
        weight_decay (float|WeightDecayRegularizer, optional): The strategy of regularization. \
            It canbe a float value as coeff of L2 regularization or \
            :ref:`api_fluid_regularizer_L1Decay`, :ref:`api_fluid_regularizer_L2Decay`.
            If a parameter has set regularizer using :ref:`api_fluid_ParamAttr` already, \
            the regularization setting here in optimizer will be ignored for this parameter. \
            Otherwise, the regularization setting here in optimizer will take effect. \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three cliping strategies
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` ,
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): The default value is None. Normally there is no need for user
                to set this property. For more information, please refer to
                :ref:`api_guide_Name` .

    Examples:
        .. code-block:: python

            import paddle

            inp = paddle.uniform([10, 10], dtype="float32", min=-0.1, max=0.1)
            linear = paddle.nn.Linear(10, 10)
            out = linear(inp)
            loss = paddle.mean(out)
            beta1 = paddle.to_tensor([0.9], dtype="float32")
            beta2 = paddle.to_tensor([0.99], dtype="float32")
            adadelta = paddle.optimizer.Adadelta(learning_rate=0.1, parameters=linear.parameters(), weight_decay=0.01)
            back = out.backward()
            adadelta.step()
            adadelta.clear_grad()

            #Note that the learning_rate of linear_2 is 0.01.
            linear_1 = paddle.nn.Linear(10, 10)
            linear_2 = paddle.nn.Linear(10, 10)
            inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
            out = linear_1(inp)
            out = linear_2(out)
            loss = paddle.mean(out)
            adadelta = paddle.optimizer.Adadelta(
                learning_rate=0.1,
                parameters=[{
                    'params': linear_1.parameters()
                }, {
                    'params': linear_2.parameters(),
                    'weight_decay': 0.001,
                    'learning_rate': 0.1,
                }],
                weight_decay=0.01)
            out.backward()
            adadelta.step()
            adadelta.clear_grad()

    """

    _avg_squared_grad_acc_str = "_avg_squared_grad"
    _avg_squared_update_acc_str = "_avg_squared_update"

    def __init__(
        self,
        learning_rate=0.001,
        epsilon=1.0e-6,
        rho=0.95,
        parameters=None,
        weight_decay=None,
        grad_clip=None,
        name=None,
    ):
        if learning_rate is None:
            raise ValueError("learning_rate is not set.")
        if epsilon is None:
            raise ValueError("epsilon is not set.")
        if rho is None:
            raise ValueError("rho is not set.")
        super().__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name,
        )
        self._multi_precision = False
        self._master_weights = {}
        self.type = "adadelta"
        self._epsilon = epsilon
        self._rho = rho
        self._default_dict = {
            'epsilon': epsilon,
            'rho': rho,
        }

    def _create_master_weight(self, param):
        if param.name in self._master_weights:
            var = self._master_weights[param.name]
        else:
            assert isinstance(self.helper, LayerHelper)

            var_name = param.name + "_fp32_master"
            var_name = unique_name.generate(var_name)
            var = paddle.static.create_global_var(
                name=var_name,
                shape=param.shape,
                value=0,
                dtype='float32',
                persistable=True,
            )
            block = self.helper.startup_program.global_block()
            block.append_op(
                type="cast",
                inputs={"X": [param]},
                outputs={"Out": [var]},
                attrs={
                    "in_dtype": param.dtype,
                    "out_dtype": core.VarDesc.VarType.FP32,
                },
            )
            self._master_weights[param.name] = var
        return var

    def _get_accumulator(self, name, param):
        """Utility function to fetch an accumulator for a parameter
        Args:
            name: name of the accumulator
            param: parameter variable for which accumulator is to be fetched
        Returns:
            accumulator variable for the parameter
        """
        if self._name is not None:
            name = self._name + "_" + name
        find_master = (
            self._multi_precision and param.dtype == core.VarDesc.VarType.FP16
        )
        target_param = (
            self._master_weights[param.name] if find_master else param
        )
        target_name = target_param.name
        if (
            name not in self._accumulators
            or target_name not in self._accumulators[name]
        ):
            raise Exception(
                "Accumulator {} does not exist for parameter {}".format(
                    name, target_name
                )
            )
        return self._accumulators[name][target_name]

    def _create_accumulators(self, block, parameters):
        if not isinstance(block, framework.Block):
            raise TypeError("block is not instance of framework.Block.")
        if isinstance(parameters, dict):
            parameters = parameters.get('params')

        for p in parameters:
            if self._multi_precision and p.dtype == core.VarDesc.VarType.FP16:
                master_p = self._create_master_weight(p)
                self._add_accumulator(self._avg_squared_grad_acc_str, master_p)
                self._add_accumulator(
                    self._avg_squared_update_acc_str, master_p
                )
                continue
            if (
                p.dtype == core.VarDesc.VarType.FP16
                and not self._multi_precision
            ):
                warnings.warn(
                    "Accumulating with FP16 in optimizer can lead to poor accuracy or slow convergence."
                    "Consider using multi_precision=True option of the Lars optimizer."
                )
            self._add_accumulator(self._avg_squared_grad_acc_str, p)
            self._add_accumulator(self._avg_squared_update_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)

        avg_squared_grad_acc = self._get_accumulator(
            self._avg_squared_grad_acc_str, param_and_grad[0]
        )
        avg_squared_update_acc = self._get_accumulator(
            self._avg_squared_update_acc_str, param_and_grad[0]
        )
        find_master = (
            self._multi_precision
            and param_and_grad[0].dtype == core.VarDesc.VarType.FP16
        )
        master_weight = (
            self._master_weights[param_and_grad[0].name]
            if find_master
            else None
        )

        if in_dygraph_mode():
            with no_grad():
                _C_ops.adadelta_(
                    param_and_grad[0],
                    param_and_grad[1],
                    avg_squared_grad_acc,
                    avg_squared_update_acc,
                    master_weight,
                    self._rho,
                    self._epsilon,
                    find_master,
                )
            return None
        else:
            if not isinstance(block, framework.Block):
                raise TypeError("block is not instance of framework.Block.")

            # Create the adadelta optimizer op
            inputs = {
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "AvgSquaredGrad": avg_squared_grad_acc,
                "AvgSquaredUpdate": avg_squared_update_acc,
            }
            outputs = {
                "ParamOut": param_and_grad[0],
                "AvgSquaredGradOut": avg_squared_grad_acc,
                "AvgSquaredUpdateOut": avg_squared_update_acc,
            }
            if find_master:
                inputs["MasterParam"] = master_weight
                outputs["MasterParamOut"] = master_weight
            adadelta_op = block.append_op(
                type=self.type,
                inputs=inputs,
                outputs=outputs,
                attrs={
                    "epsilon": self._epsilon,
                    "rho": self._rho,
                    "multi_precision": find_master,
                },
                stop_gradient=True,
            )

            return adadelta_op

    def _update_param_group(self, parameters):
        self._epsilon = parameters.get('epsilon', self._default_dict['epsilon'])
        self._rho = parameters.get('rho', self._default_dict['rho'])
        parameters = parameters.get('params')
        return parameters
