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

import inspect
import paddle
try:
    from .proto import pass_desc_pb2
except ModuleNotFoundError:
    import sys
    from .proto import framework_pb2
    sys.path.append(framework_pb2.__file__.rsplit('/', 1)[0])
    from .proto import pass_desc_pb2
from .core import register_pass

ALL_REGISTER_PASS = dict()


class Attr(object):
    def __init__(self, name):
        self._name = name
        self._value = None
        self._is_mapped = False
        self._mapped_op = None
        self._mapped_attr = None

    def Set(self, value):
        self._value = value

    def Reuse(self, op_type, attr_name):
        self._is_mapped = True
        self._mapped_op = op_type
        self._mapped_attr = attr_name

    def IsMapped(self):
        return self._is_mapped

    def ToOpDescAttr(self):
        attr = framework_pb2.OpDesc.Attr()
        return attr


class Var(object):
    def __init__(self, name):
        self._name = name
        self._op_type = None
        self._is_belong_to_op = False

    def Name(self):
        if self._is_belong_to_op:
            return "%s.%s" % (self._op_type, self._name)
        else:
            return self._name

    def OpType(self):
        return self._op_type

    def SetOpType(self, op_type):
        self._is_belong_to_op = True
        self._op_type = op_type

    def SetName(self, name):
        self._name = name


class Op(object):
    def __init__(self, type):
        self._type = type
        self._attrs = list()
        self._inputs = dict()
        self._outputs = dict()

    def Type(self):
        return self._type

    def Attr(self, attr_name):
        attr = Attr(attr_name)
        self._attrs.append(attr)
        return attr

    def Attrs(self):
        return self._attrs

    def _set_vars(self, var_maps, kwargs, belong_to_op=False):
        for (parameter, vars) in kwargs.items():
            if isinstance(vars, Var):
                vars = [vars]
            for var in vars:
                if belong_to_op:
                    var.SetOpType(self._type)
            var_maps[parameter] = vars

    def SetInput(self, **kwargs):
        self._set_vars(self._inputs, kwargs)
        return self

    def SetOutput(self, **kwargs):
        self._set_vars(self._outputs, kwargs, True)
        return self

    def Inputs(self):
        return self._inputs

    def Output(self, out_name):
        return self._outputs.get(out_name)

    def Outputs(self):
        return self._outputs

    def ToOpDesc(self, op_desc):
        def _var_to_desc_var(desc_vars, var_maps):
            for parameter, vars in var_maps.items():
                desc_var = desc_vars.add()
                desc_var.parameter = parameter
                for var in vars:
                    desc_var.arguments.append(var.Name())

        op_desc.type = self._type
        _var_to_desc_var(op_desc.inputs, self._inputs)
        _var_to_desc_var(op_desc.outputs, self._outputs)
        return op_desc


class DescribeFunctionHelper(object):
    def __init__(self, func):
        self._func = func
        self._vars = None
        self._ops = None

    def Vars(self):
        return list(map(lambda var: var.Name(), self._vars))

    def Build(self, input_specs):
        arg_specs = inspect.getfullargspec(self._func)
        self._vars = list(map(Var, arg_specs.args))
        ops = self._func(*self._vars)
        if isinstance(ops, (list, tuple)):
            self._ops = ops
        else:
            self._ops = [ops]
        return self

    def ToProgramDesc(self, program_desc):
        block_desc = program_desc.blocks.add()
        block_desc.idx = 0
        block_desc.parent_idx = 0
        for op in self._ops:
            op.ToOpDesc(block_desc.ops.add())

    def ToAttrMap(self, attr_maps):
        for io in self._ops:
            pass


class APIFunctionHelper(object):
    def __init__(self, func):
        self._func = func
        self._vars = None
        self._program = None

    def Vars(self):
        vars = list()
        for var in self._vars:
            vars.append(var.name)
        return vars

    def _get_args_from_func(self, func, input_specs):
        args = list()
        arg_specs = inspect.getfullargspec(self._func)
        for arg_name in arg_specs.args:
            input_spec = input_specs.get(arg_name)
            if isinstance(input_spec, paddle.static.InputSpec):
                args.append(
                    paddle.static.data(arg_name, input_spec.shape,
                                       input_spec.dtype))
            elif isinstance(input_spec, paddle.ParamAttr):
                args.append(paddle.ParamAttr(arg_name))
            else:
                args.append(paddle.static.data(arg_name, [-1]))
        return args

    def Build(self, input_specs):
        switch_static_mode = paddle.in_dynamic_mode()
        if switch_static_mode:
            paddle.enable_static()
        self._program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(self._program, startup_program):
            self._vars = self._get_args_from_func(self._func, input_specs)
            self._vars.append(self._func(*self._vars))
        if switch_static_mode:
            paddle.disable_static()
        return self

    def ToProgramDesc(self, program_desc):
        program_desc.ParseFromString(self._program.desc.serialize_to_string())


class RegisterPassHelper(object):
    def __init__(self, pass_name):
        self._name = pass_name
        self._pass_pairs = None
        self._input_specs = dict()

    def SetPassPairs(self, pass_pairs):
        self._pass_pairs = pass_pairs

    def SetInputSpecs(self, input_specs):
        if isinstance(input_specs, dict):
            self._input_specs.update(input_specs)

    def GetMultiPassDesc(self):
        assert self._pass_pairs
        multi_pass_desc = pass_desc_pb2.MultiPassDesc()
        multi_pass_desc.name = self._name
        for (pattern, replace) in self._pass_pairs:
            pass_desc = multi_pass_desc.pass_descs.add()
            # pattern
            pattern.Build(self._input_specs).ToProgramDesc(pass_desc.pattern)
            # replace
            replace.Build(self._input_specs).ToProgramDesc(pass_desc.replace)
            # var map
            pattern_vars = pattern.Vars()
            replace_vars = replace.Vars()
            assert len(pattern_vars) == len(replace_vars)
            for (pattern_var, replace_var) in zip(pattern_vars, replace_vars):
                var_map = pass_desc.var_maps.add()
                var_map.pattern_var = pattern_var
                var_map.replace_var = replace_var
            # attr map
            if isinstance(replace, DescribeFunctionHelper):
                replace.ToAttrMap(pass_desc.attr_maps)
        return multi_pass_desc


def RegisterPass(pass_name,
                 pattern_use_api=True,
                 replace_use_api=True,
                 input_specs=None):
    pattern_helper_class = APIFunctionHelper if pattern_use_api else DescribeFunctionHelper
    replace_helper_class = APIFunctionHelper if replace_use_api else DescribeFunctionHelper

    def _func_to_helper(pass_pair_func):
        pattern, replace = pass_pair_func
        return pattern_helper_class(pattern), replace_helper_class(replace)

    def _is_pass_pair(check_pair):
        if isinstance(check_pair, (list, tuple)):
            if len(check_pair) == 2:
                if all(map(inspect.isfunction, check_pair)):
                    return True
        return False

    def _register_pass_warpper(func):
        register_helper = RegisterPassHelper(pass_name)
        pass_pairs = list()
        pass_pair_funcs = func()
        if _is_pass_pair(pass_pair_funcs):
            pass_pairs.append(_func_to_helper(pass_pair_funcs))
        elif all(map(_is_pass_pair, pass_pair_funcs)):
            pass_pairs.extend(map(_func_to_helper, pass_pair_funcs))
        else:
            raise ValueError("Error Pass Pair functions.")
        register_helper.SetPassPairs(pass_pairs)
        register_helper.SetInputSpecs(input_specs)
        desc = register_helper.GetMultiPassDesc()
        register_pass(pass_name, desc.SerializeToString())
        return func

    return _register_pass_warpper
