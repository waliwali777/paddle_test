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

import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Dict, Any

import paddle
from paddle.framework import core
from paddle.fluid.framework import program_guard, device_guard
from paddle.fluid import unique_name, layers
from paddle.fluid.clip import append_gradient_clip_ops
from .pass_base import PassBase, PassType, register_pass

GRAD_MERGE_COND_NAME = "grad_merge_cond_name"


def _is_the_backward_op(op):
    OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()
    OpRole = core.op_proto_and_checker_maker.OpRole
    return OP_ROLE_KEY in op.attr_names and \
            int(op.all_attrs()[OP_ROLE_KEY]) & int(OpRole.Backward)


def _is_the_optimizer_op(op):
    OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()
    OpRole = core.op_proto_and_checker_maker.OpRole
    return OP_ROLE_KEY in op.attr_names and \
            int(op.all_attrs()[OP_ROLE_KEY]) & int(OpRole.Optimize)


def _remove_and_get_optimizer_op(main_program, dist_context):
    # 1 create tmp block
    # 2 mv optimizer op from global program to tmp block
    # 3 del the op from dist_context
    from paddle.distributed.fleet.meta_optimizers.common import OpRole
    main_block = main_program.global_block()
    temp_block = main_program._create_block()
    removed_op_idx = []
    optimize_ops_desc = []
    for idx, op in enumerate(main_block.ops):
        if _is_the_optimizer_op(op):
            # append optimizer op to tmp block
            new_op_desc = temp_block.desc.append_op()
            new_op_desc.copy_from(op.desc)
            optimize_ops_desc.append(new_op_desc)
            removed_op_idx.append(idx)

            # del op from dist_context
            if dist_context:
                dist_context.del_dist_op_for_program(op)

    for idx in removed_op_idx[::-1]:
        main_block._remove_op(idx)

    return optimize_ops_desc


def _remove_op_role_var(param, grad):
    op_maker = core.op_proto_and_checker_maker
    op = grad.op
    assert _is_the_backward_op(op), \
        'grad.op={} is not the backward op which produces the grad={}' \
        .format(op, grad.name)

    block = grad.block
    var_attr = op.all_attrs()[op_maker.kOpRoleVarAttrName()]
    if len(var_attr) == 0:
        return

    assert param.name in var_attr, \
        'when using GradientMergeOptimizer, param={} must be in var_attr={}' \
        .format(param.name, var_attr)
    assert grad.name in var_attr, \
        'when using GradientMergeOptimizer, grad={} must be in var_attr={}' \
        .format(param.name, var_attr)

    # remove (param, grad) from op_role_var
    var_attr.remove(param.name)
    var_attr.remove(grad.name)
    if len(var_attr) > 1:
        op._set_attr(op_maker.kOpRoleVarAttrName(), var_attr)
    else:
        op._remove_attr(op_maker.kOpRoleVarAttrName())


def _add_gm_op_role_var(op, param, grad, cond_var_name):
    grad.op = op
    op_maker = core.op_proto_and_checker_maker
    backward = op_maker.OpRole.Backward

    # NOTE(wangxi). When distributed, we will insert grad_merge_all_reduce_op_handle
    # in multi_devices_graph_pass, which will allreduce(grad) if cond is True, else
    # do nothing.
    # In this way, the gradient can be merged first, and then communicate when the
    # condition is met, reducing the number of communications to increase the
    # speed.
    op._set_attr(GRAD_MERGE_COND_NAME, cond_var_name)
    op._set_attr(op_maker.kOpRoleAttrName(), backward)
    op._set_attr(op_maker.kOpRoleVarAttrName(), [param.name, grad.name])


def _get_gm_cond_var(main_program, k_steps):
    main_block = main_program.global_block()
    # Add const var
    k_step_var = layers.create_global_var(
        name="gradient_merge_k",
        shape=[1],
        value=int(k_steps),
        dtype='int32',
        persistable=True,
        force_cpu=True)

    zero_var = layers.create_global_var(
        name="gradient_merge_zero",
        shape=[1],
        value=int(0),
        dtype='int32',
        persistable=True,
        force_cpu=True)

    # Add step var & cond var
    step_var = layers.create_global_var(
        name="gradient_merge_step",
        shape=[1],
        value=int(0),
        dtype='int32',
        persistable=True,
        force_cpu=True)

    cond_var = layers.create_global_var(
        name="gradient_merge_cond",
        shape=[1],
        value=bool(0),
        dtype='bool',
        persistable=False,
        force_cpu=True)

    with device_guard("cpu"):
        # step_var = (step_var + 1) % k_step
        layers.increment(x=step_var, value=1.0, in_place=True)
        main_block.append_op(
            type='elementwise_mod',
            inputs={'X': step_var,
                    'Y': k_step_var},
            outputs={'Out': step_var},
            attrs={'axis': -1,
                   'use_mkldnn': False})

        # cond_var = (step_var == 0)
        main_block.append_op(
            type='equal',
            inputs={'X': step_var,
                    'Y': zero_var},
            outputs={'Out': cond_var})

    return cond_var


def _append_gradient_merge_backward_op(
        main_program,
        startup_program,
        params_grads: List[Tuple[Any, Any]],
        cond_var_name: str) -> Tuple[List[Tuple[Any, Any]], Dict[str, Any]]:
    main_block = main_program.global_block()
    startup_block = startup_program.global_block()

    #TODO(mapingshuo) support sparse embedding
    # step1: remove grad.op's op_role_var
    for param, grad in params_grads:
        assert (
            param.type != core.VarDesc.VarType.SELECTED_ROWS
        ), "SELECTED_ROWS is not supported in GradientMergeOptimizer for now"

        _remove_op_role_var(param, grad)

    param_gradientmerge_var_map = {}
    params_grad_list = []
    # step2: create gradient_merge var and init with 0
    # and update op_role_var
    for param, grad in params_grads:
        param_name = param.name
        param_var = main_block.var(param_name)
        assert (param_var is not None)
        gradient_merge_var = main_block.create_var(
            name=param_name + "@GRAD@GradientMerge",
            shape=param_var.shape,
            dtype=param_var.dtype,
            persistable=True)
        param_gradientmerge_var_map[param_name] = gradient_merge_var

        startup_gradient_merge_var = startup_block.create_var(
            name=param_name + "@GRAD@GradientMerge",
            shape=param_var.shape,
            dtype=param_var.dtype,
            persistable=True)
        startup_block.append_op(
            type="fill_constant",
            outputs={"Out": startup_gradient_merge_var},
            attrs={
                "shape": param_var.shape,
                "dtype": param_var.dtype,
                "value": float(0),
            })

        # grad_merge += grad
        new_grad_op = main_block.append_op(
            type="elementwise_add",
            inputs={'X': grad,
                    'Y': gradient_merge_var},
            outputs={'Out': gradient_merge_var},
            attrs={'axis': -1,
                   'use_mkldnn': False})
        _add_gm_op_role_var(new_grad_op, param, gradient_merge_var,
                            cond_var_name)
        params_grad_list.append([param, gradient_merge_var])
    return params_grad_list, param_gradientmerge_var_map


def _create_cond_block_and_update_optimizer(
        main_program,
        cond_var,
        new_params_grads: List[Tuple[Any, Any]],
        param_gradientmerge_var_map: Dict[str, Any],
        optimize_ops_desc: List[Any],
        k_steps,
        avg):
    def true_apply_gradient():
        cur_block_idx = main_program.current_block_idx
        cur_block = main_program.current_block()

        # cur_block's forward_block & backward_block is itself
        cur_block._set_forward_block_idx(cur_block_idx)
        op_maker = core.op_proto_and_checker_maker
        if avg:
            for param, new_grad in new_params_grads:
                # grad /= k_steps
                cur_block.append_op(
                    type='scale',
                    inputs={'X': new_grad},
                    outputs={'Out': new_grad},
                    attrs={
                        'scale': 1.0 / k_steps,
                        'bias': 0.0,
                        'bias_after_scale': False
                    })
                new_grad.op._set_attr(op_maker.kOpRoleAttrName(),
                                      op_maker.OpRole.Optimize)

        # append optimizer ops
        for op_desc in optimize_ops_desc:
            new_op_desc = cur_block.desc.append_op()
            new_op_desc.copy_from(op_desc)

            #update input/output
            for input_name in new_op_desc.input_arg_names():
                if input_name in new_params_grads:
                    new_op_desc._rename_input(input_name,
                                              new_params_grads[input_name])

            for output_name in new_op_desc.output_arg_names():
                if output_name in new_params_grads:
                    new_op_desc._rename_output(output_name,
                                               new_params_grads[output_name])

            # update op_role_var
            if new_op_desc.has_attr(op_maker.kOpRoleVarAttrName()):
                var_attr = new_op_desc.attr(op_maker.kOpRoleVarAttrName())
                param_name = var_attr[0]
                grad_var = param_gradientmerge_var_map[param_name]
                new_op_desc._set_attr(op_maker.kOpRoleVarAttrName(),
                                      [param_name, grad_var.name])

            # op's update Grad
            if new_op_desc.input("Grad"):
                grad_value = new_op_desc.input("Grad")[0]
                # TODO FIXME(xym) support fp16
                grad_merge_value = grad_value + '@GradientMerge'
                new_op_desc.set_input("Grad", [grad_merge_value])

        main_program.global_block()._sync_with_cpp()
        cur_block._sync_with_cpp()

        # clear gradient_merge_vars
        for param, new_grad in new_params_grads:
            layers.fill_constant(
                shape=new_grad.shape,
                dtype=new_grad.dtype,
                value=0.0,
                out=new_grad)
            new_grad.op._set_attr(op_maker.kOpRoleAttrName(),
                                  op_maker.OpRole.Optimize)

    layers.cond(cond_var, true_fn=true_apply_gradient, false_fn=None)


def parse_program(main_program, startup_program, params_grads, k_steps, avg,
                  dist_context):
    # 1 create gradient_merge_cond
    cond_var = _get_gm_cond_var(main_program, k_steps)

    # 2 remove optimizer_op from main_program
    optimize_ops_desc = _remove_and_get_optimizer_op(main_program, dist_context)

    # back to block 0
    main_program._rollback()

    # 3 append gradient merge backward op to main_program
    new_params_grads, param_gradientmerge_var_map = _append_gradient_merge_backward_op(
        main_program, startup_program, params_grads, cond_var.name)

    # 4 create ConditionalBlock and append gradient merge optimizer ops
    _create_cond_block_and_update_optimizer(
        main_program, cond_var, new_params_grads, param_gradientmerge_var_map,
        optimize_ops_desc, k_steps, avg)


@register_pass("auto_parallel_gradient_merge_pass")
class GradientMergePass(PassBase):
    def __init__(self):
        super(GradientMergePass, self).__init__()
        self.set_attr("k_steps", -1)
        self.set_attr("avg", True)
        self.set_attr("inner_optimizer", None)

    def _check_self(self):
        if self.get_attr("k_steps") < 1:
            return False
        return True

    def _check_conflict(self, other_pass):
        return True

    def _type(self):
        return PassType.COMM_OPT

    def _apply_single_impl(self, main_program, startup_program, context):
        k_steps = self.get_attr("k_steps", -1)
        avg = self.get_attr("avg", False)
        dist_context = self.get_attr("dist_context")
        params_grads = self.get_attr("params_grads")
        with paddle.static.program_guard(main_program, startup_program):
            parse_program(main_program, startup_program, params_grads, k_steps,
                          avg, dist_context)

        main_program._sync_with_cpp()
