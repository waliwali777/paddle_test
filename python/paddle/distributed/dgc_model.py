# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except jin compliance with the License.
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
import six
import numpy as np
import warnings
from collections import OrderedDict,defaultdict
import itertools
import warnings
from functools import reduce

import paddle
from paddle.fluid import core
from paddle.fluid import framework
from paddle.fluid.dygraph import layers
from paddle.fluid.dygraph import parallel_helper
from paddle.fluid.dygraph import to_variable, no_grad
from paddle.regularizer import L1Decay, L2Decay
from paddle.utils import deprecated, unique_name
from paddle.fluid.layers import collective
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph import base as imperative_base 
from paddle.nn.initializer import Constant
from paddle.fluid.layers import tensor
from paddle import _C_ops
from paddle.fluid.framework import ParamBase, _in_legacy_dygraph, in_dygraph_mode, device_guard

__all__ = ["dgc_data_parallel"]

def dgc_data_parallel(model, optimizer, rampup_begin_step=10, rampup_step=1, sparsity=[0.999]):
    dgc_model = DGCModel(layers=model,
                         learning_rate=optimizer._learning_rate,
                         momentum=optimizer._momentum,
                         rampup_begin_step=rampup_begin_step,
                         rampup_step=rampup_step,
                         sparsity=sparsity,
                         use_nesterov=optimizer._use_nesterov,
                         weight_decay=optimizer._regularization_coeff,
                         grad_clip=optimizer._grad_clip)

    dgc_optimizer = DGCDygraphOptimizer(dgc_model)

    return dgc_model, dgc_optimizer


ParallelStrategy = core.ParallelStrategy
ParallelEnv = paddle.fluid.dygraph.ParallelEnv

def _build_default_parallel_strategy():
    strategy = ParallelStrategy()
    strategy.nranks = ParallelEnv().nranks
    strategy.local_rank = ParallelEnv().local_rank
    strategy.trainer_endpoints = ParallelEnv().trainer_endpoints
    strategy.current_endpoint = ParallelEnv().current_endpoint
    return strategy

@imperative_base.no_grad
@framework.dygraph_only
def sync_params_buffers(params,
                        group=None,
                        src_rank=0):
    
    if len(params) == 0:
        return

    for param in params:
        if not isinstance(param, core.VarBase):
            raise TypeError("The data type of '%s' must be VarBase" %
                            param.name)

    for param in params:
        paddle.distributed.broadcast(
            param, src=src_rank, group=group, use_calc_stream=True)

class DGCModel(layers.Layer):

    def __init__(self,
                 layers,
                 learning_rate=0.01,
                 momentum=0.9,
                 rampup_begin_step=10,
                 rampup_step=1,
                 sparsity=[0.999],
                 use_nesterov=False,
                 strategy=None,
                 weight_decay=None,
                 grad_clip=None,
                 group=None,
                 find_unused_parameters=False):
        super(DGCModel,
              self).__init__(layers.full_name() + "_dgc_data_parallel")

        self._layers = layers
        self._momentum = momentum
        self._use_nesterov = use_nesterov
        self._grad_clip = grad_clip
        self.trainable_params = []
        self.reduce_order = 0
        self.reduce_count = -1
        self._dgc_clip_norm = None
        self._learning_rate = learning_rate
        self._global_learning_rate = paddle.to_tensor(learning_rate, dtype="float32")
        self._weight_decay = weight_decay
        self._u_velocity_acc_str = "_dgc_u_"
        self._v_velocity_acc_str = "_dgc_v_"
        self._dgc_vars = defaultdict(lambda: list())
        self._reduce_var_ready = {}
        self._reduce_var_index = {}
        self.helper = LayerHelper(self.__class__.__name__)

        ranks = list(range(paddle.distributed.get_world_size())) if group is None else group.ranks
        self._num_trainers = len(ranks)

        # reuse some attrs in dgc_op and dgc_momentum_op
        self._accumulators = defaultdict(lambda: dict())
        self._opti_name_list = []

        if grad_clip is not None:
            if not isinstance(grad_clip, paddle.nn.ClipGradByNorm):
                raise TypeError(
                    "The type of grad_clip should be 'ClipGradByNorm', because DGCModel only support ClipGradByNorm"
                )
           
            assert len(ranks) > 0, "dgc only works on multi cards."

            self._dgc_clip_norm = grad_clip.clip_norm * (self._num_trainers**-0.5)

        self.regular_type, self.regular_coeff = self._get_regularization_param(weight_decay)

        self.new_group = paddle.distributed.new_group(ranks)

        # NOTE: The ParallelStrategy here is not strictly a strategy. 
        # It just stores some environment variables, which can be constructed by 
        # ParallelEnv. Here it is set as an optional argument.
        # This parameter is not removed because of compatibility with 1.x writing.
        if strategy is not None:
            self._strategy = strategy
        else:
            self._strategy = _build_default_parallel_strategy()
        
        # dgc config
        self._rampup_begin_step = rampup_begin_step
        self._rampup_step = rampup_step
        self._sparsity = sparsity

        if len(ranks) > 1:
            # check the environment
            assert parallel_helper.__parallel_ctx__clz__ is not None, \
            "ParallelContext must be initialized before. You should use init_parallel_env() before" \
            "constructing the DataParallel."
    
            # sync buffer and params
            sync_params_buffers(layers.parameters(), group=self.new_group)

            self.init_params()
            self.register_bw_hooks()
        else:
            raise Exception("dgc only works on multi cards.")
    
    def _get_regularization_param(self, regularization):
        regular_type = 0
        regular_coeff = 0.0

        if regularization is not None:
            if isinstance(regularization, float):
                regular_type = 2
                regular_coeff = regularization
                return regular_type, regular_coeff
            else:
                regular_coeff = regularization._regularization_coeff
                if isinstance(regularization, L1Decay):
                    regular_type = 1
                elif isinstance(regularization, L2Decay):
                    regular_type = 2
                else:
                    assert False, 'regularization must be None|L1Decay|L2Deacy'

        return regular_type, regular_coeff

    def forward(self, *inputs, **kwargs):
        self._clear_counters()

        outputs = self._layers(*inputs, **kwargs)
        return outputs
    
    def register_bw_hooks(self):
        for param in self.trainable_params:
            #NOTE: clip grad and add dgc op in backward hook.
            reduce_function = self._get_reduce_fn(param)
            param.register_hook(reduce_function)
            #param._register_backward_hook(reduce_function)

    def _get_reduce_fn(self, param):
        # Buffer reduction
        @imperative_base.no_grad
        def reduce_grad(grad):
            nonlocal param
            #NOTE: 1:ready, 0: not ready
            self._reduce_var_ready[param.name] = 1
            
            assert grad is not None, "Parameter gradient cannot be None"
            
            # comm after div
            grad = grad.scale_(1.0 / self.new_group.nranks)
            
            if self._is_use_dgc(param, grad):
                clip_var = grad
                
                # local gradient clipping
                if self._global_step_var[0] >= self._rampup_begin_step and self._dgc_clip_norm is not None:
                    clip_var = self._clip_by_norm(grad, max_norm=self._dgc_clip_norm, name=param.name + "_grad")
                
                u_var, v_var = self._accumulators[self._u_velocity_acc_str][param.name], self._accumulators[self._v_velocity_acc_str][param.name]
                
                k_var, encoded_var, gather_var = self._get_auxiliary_var(param)
                
                
                # momentum correction and local regularization
                self._dgc_op(param, clip_var, grad, u_var, v_var, k_var,
                                encoded_var, gather_var)
                k_var[0] = encoded_var.shape[0] // 2

            def reduce_buffer(comm_param, comm_grad):
                assert comm_grad is not None
                paddle.distributed.wait(comm_grad, group=self.new_group, use_calc_stream=True)
                #NOTE: dgc comm: allgather.
                if self._global_step_var[0] >= self._rampup_begin_step and  self._is_use_dgc(comm_param, comm_grad):
                    k_var, comm_encoded_var, comm_gather_var = self._dgc_vars[comm_param.name]
                    comm_grad = paddle.distributed.utils.dgc_comm(comm_encoded_var, comm_gather_var, comm_grad, int(k_var[0]), self.new_group)
                else:
                    with paddle.no_grad():
                        paddle.distributed.all_reduce(comm_grad, group=self.new_group, use_calc_stream=False)

                paddle.distributed.wait(comm_grad, group=self.new_group, use_calc_stream=False)
                paddle.assign(comm_grad, comm_param.grad)
                
                self.reduce_order += 1

                return comm_grad

            res_grad = reduce_buffer(param, grad)

            return res_grad

        return reduce_grad

    def _dgc_op(self, param_var, clip_var, grad_var, u_var, v_var, k_var,
            encoded_var, gather_var):

        regular_type = self.regular_type
        regular_coeff = self.regular_coeff
        # The regularizer of the Parameters have higher priority
        if param_var.regularizer is not None:
            regular_type, regular_coeff = self._get_regularization_param(
                param_var.regularizer)
        
        with paddle.no_grad():
            paddle.fluid.framework._dygraph_tracer().trace_op(
                                    type="dgc",
                                    inputs={
                                        "U": u_var,
                                        "V": v_var,
                                        "Grad": clip_var,
                                        "Param": param_var,
                                        "current_step": self._global_step_var.clone().cpu(),
                                        "nranks": self._nranks_var.cpu(),
                                    },
                                    outputs={
                                        "U_out": u_var,
                                        "V_out": v_var,
                                        "EncodeGrad": encoded_var,
                                        "k": k_var.cpu(),
                                        "Grad_out": grad_var,
                                        "GatherBuff": gather_var,
                                    },
                                    attrs={
                                        "m": self._momentum,
                                        "sparsity": self._sparsity,
                                        "use_nesterov": self._use_nesterov,
                                        "rampup_begin_step": float(self._rampup_begin_step),
                                        "rampup_step": float(self._rampup_step),
                                        "regular_coeff": float(regular_coeff),
                                        "regular_type": int(regular_type),
                                    })
            
    def _clip_by_norm(self, x, max_norm, name=None):
        args = {'x': x, 'max_norm': max_norm, 'name': name}

        helper = LayerHelper("dgc_clip_by_norm_op", **args)

        if name is None:
            name = unique_name.generate_with_ignorable_key(".".join(
                [helper.name, 'tmp']))

        out = helper.create_variable(
            type=x.type, name=name, dtype=x.dtype, persistable=False)
        with paddle.no_grad():
            paddle.fluid.framework._dygraph_tracer().trace_op(
                                    type="dgc_clip_by_norm",
                                    inputs={"X": x,
                                            "current_step": self._global_step_var.clone().cpu()},
                                    attrs={
                                        "max_norm": max_norm,
                                        "rampup_begin_step": float(self._rampup_begin_step)
                                    },
                                    outputs={"Out": out})
        return out
    
    def _create_regularization_of_grad(self, param, grad):
        """ Create and add backward regularization Operators
    
        Function helper of append_regularization_ops.
        """
        if isinstance(self._weight_decay, float):
            regularization = L2Decay(self._weight_decay)
        else:
            regularization = self._weight_decay

        # If no gradient or no regularization is specified,  then we don't need to do anything
        if grad is None or ((not hasattr(param, 'regularizer') or
                             (hasattr(param, 'regularizer') and
                              param.regularizer is None)) and
                            regularization is None):
            return grad
        regularization_term = None
        if hasattr(param, 'regularizer') and param.regularizer is not None:
            # Add variable for regularization term in grad block
            regularization_term = param.regularizer(param, grad, grad.block)
        elif regularization is not None:
            regularization_term = regularization(param, grad, grad.block)

        assert regularization_term is not None

        return _C_ops.sum([grad, regularization_term])
        
    def init_params(self):
        layers_param = []
        params_set = set()
        for sublayer in self.sublayers():
            for _, param in sublayer.named_parameters(include_sublayers=False):
                if param is None or param in params_set:
                    continue
                params_set.add(param)
                if not isinstance(param, core.VarBase):
                    raise TypeError("The data type of '%s' must be Varbase" %
                                    param.name)
                if param.trainable:
                    layers_param.append((sublayer, param))

        self.trainable_params = [param for _, param in layers_param]
        num_trainable_params = len(self.trainable_params)
        for index, (_, param) in enumerate(layers_param):
            self._reduce_var_ready[param.name] = 0
            self._reduce_var_index[param.name] = num_trainable_params - index - 1
        
        self.trainable_params.reverse()

        assert len(self.trainable_params) == len(self._reduce_var_ready), "there are at least two params with same name."

        assert len( self.trainable_params) > 0, \
            "This model does not have any parameters to train, and " \
            "does not need to use DataParallel"
        
        self.reduce_count = len(self.trainable_params)

        # step counter
        self._global_step_var = paddle.to_tensor([0], dtype='float32')

        self._nranks_var = paddle.to_tensor([self._num_trainers], dtype='float32')

        # rampup begin step var for all_reduce_op_handle
        self._rampup_begin_step_var = tensor.create_global_var(
            shape=[1],
            dtype=core.VarDesc.VarType.FP32,
            persistable=True,
            name=core.dgc.kDGCRampUpBeginStepName(),
            value=self._rampup_begin_step * 1.0,
            force_cpu=True)
        
        for param_var in self.trainable_params:
            # reuse velocity in dgc_op and dgc_momentum_op
            self._add_accumulator(self._u_velocity_acc_str, param_var)

            if not self._is_use_dgc(param_var, None):
                continue

            self._add_accumulator(self._v_velocity_acc_str, param_var)       
        
    @imperative_base.no_grad
    def _clear_counters(self):
        """Reset all the grad reduce and call counters."""
        self.reduce_order = 0
        for k in self._reduce_var_ready.keys():
            self._reduce_var_ready[k] = 0
        

    def _is_use_dgc(self, param, grad):
        param_var, grad_var = param, grad
        var_numel = abs(reduce(lambda x, y: x * y, param_var.shape))
        if var_numel < 16384 or \
            param_var.type == core.VarDesc.VarType.SELECTED_ROWS  or \
            param_var.dtype != core.VarDesc.VarType.FP32 or \
            (grad_var is not None and grad_var.type == core.VarDesc.VarType.SELECTED_ROWS):
            return False
        return True

    def _get_device_for_param(self, param_name):
        device = None
        if param_name in self._param_device_map:
            device = self._param_device_map[param_name]
        return device
    
    def _get_auxiliary_var(self,param):

        if param.name in self._dgc_vars.keys():
            k_var, encoded_var, gather_var = self._dgc_vars[param.name]
            return k_var, encoded_var, gather_var

        self._dgc_vars[param.name].clear()

        k_var = paddle.to_tensor([0], dtype=param.dtype)
        encoded_var = paddle.empty(shape=[0], dtype=param.dtype, name=param.name + core.dgc.kDGCEncodedName())
        gather_var = paddle.empty(shape=[0], dtype=param.dtype,name=param.name + core.dgc.kDGCGatherName())
        
        self._dgc_vars[param.name].extend([k_var, encoded_var, gather_var])

        return k_var, encoded_var, gather_var

    def _add_accumulator(self,
                        name,
                        param,
                        dtype=None,
                        fill_value=0.0,
                        shape=None,
                        type=None,
                        device=None):
        """Utility function to add an accumulator for a parameter

        Args:
            block: the block in which the loss tensor is present
            name: name of the accumulator
            param: parameter tensor for which accumulator is to be added
            dtype: data type of the accumulator tensor
            fill_value: value to initialize the accumulator tensor
        """
        if (name in self._accumulators and
                param.name in self._accumulators[name]):
            if _in_legacy_dygraph():
                return self._accumulators[name][param.name]
            raise Exception("Accumulator {} already exists for parameter {}".
                            format(name, param.name))
        if shape == None:
            shape = param.shape
        assert isinstance(self.helper, LayerHelper)

        var_name = param.name + "_" + name
        var_name = unique_name.generate(var_name)
        self._opti_name_list.append(var_name)

        var = self.helper.create_global_variable(
            name=var_name,
            persistable=True,
            dtype=dtype or param.dtype,
            type=core.VarDesc.VarType.LOD_TENSOR
            if _in_legacy_dygraph() else (param.type
                                                if type is None else type),
            shape=shape,
            belong_to_optimizer=True)

        with device_guard(device):
            self.helper.set_variable_initializer(
                var, initializer=Constant(value=float(fill_value)))

        self._accumulators[name][param.name] = var
        return var
    
    def _create_param_lr(self, param_and_grad):
        # create learning rate tensor for every parameter
        param = param_and_grad[0]
        if hasattr(param, 'optimize_attr'):
            param_lr = param.optimize_attr['learning_rate']
            if type(param_lr) == paddle.fluid.framework.Variable:
                return param_lr
            else:
                if param_lr == 1.0:
                    return self._global_learning_rate
                else:
                    return self._global_learning_rate * param_lr
        else:
            return self._global_learning_rate
    
    def clear_grad(self, set_to_zero=True):
        core.clear_gradients(self.trainable_params, set_to_zero)

    
    @imperative_base.no_grad
    def step(self):
        """
           execute after loss.backward.
        """
        for param in self.trainable_params:
            paddle.distributed.wait(param.grad, group=self.new_group, use_calc_stream=False)
            paddle.distributed.wait(param.grad, group=self.new_group, use_calc_stream=True)

        for param in self.trainable_params:
            
            velocity_acc = self._accumulators[self._u_velocity_acc_str][param.name]
            assert velocity_acc is not None
            lr = self._create_param_lr([param, param.grad])

            if not self._is_use_dgc(param, param.grad):
                param, grad = param, param.grad
                if self._grad_clip is not None:
                    param, grad = self._grad_clip([(param, grad)])[0]
                else:
                    clip_attr = getattr(param, 'gradient_clip_attr', None)
                    assert clip_attr is None, "param does not support gradient_clip_attr in DGC now, you can globally clip in optimizer."
                #regularation
                grad = self._create_regularization_of_grad(param, grad)

                inputs = {
                    "Param": param,
                    "Grad": grad,
                    "Velocity": velocity_acc,
                    "LearningRate": lr,
                }
                outputs = {
                    "ParamOut": param,
                    "VelocityOut": velocity_acc,
                }
                attrs = {"mu": self._momentum, "use_nesterov": self._use_nesterov}
                _, _, _ = _C_ops.momentum(
                            param, grad, velocity_acc, lr,
                            (None), param, velocity_acc, (None),
                            'mu', self._momentum, 'use_nesterov', self._use_nesterov)
            else:
                opt_type = "dgc_momentum"
                inputs = {
                    "Param": param,
                    "Grad": param.grad,
                    "Velocity": velocity_acc,
                    "LearningRate": lr,
                    "current_step": self._global_step_var.clone().cpu(),
                    "nranks": self._nranks_var.cpu()
                }
                outputs = {
                    "ParamOut": param,
                    "VelocityOut": velocity_acc,
                    'Grad_out': param.grad
                }
                attrs = {"mu": self._momentum, "use_nesterov": self._use_nesterov, "rampup_begin_step": float(self._rampup_begin_step)}
        
                with paddle.no_grad():
                    paddle.fluid.framework._dygraph_tracer().trace_op(
                                                        type=opt_type,
                                                        inputs=inputs,
                                                        outputs=outputs,
                                                        attrs=attrs)

        self._global_step_var += 1

    
class DGCDygraphOptimizer(paddle.optimizer.Optimizer):
    def __init__(self, dgc_model):
        super(DGCDygraphOptimizer, self).__init__(learning_rate=dgc_model._learning_rate,
                                                parameters=dgc_model.trainable_params,
                                                weight_decay=dgc_model._weight_decay,
                                                grad_clip=dgc_model._grad_clip)
        self.dgc_model = dgc_model
    
    def step(self):
        self.dgc_model.step()
    
    def clear_grad(self):
        self.dgc_model.clear_grad()

    

