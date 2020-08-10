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

import six
import inspect
import numpy as np
import collections
from paddle.fluid import core
from paddle.fluid.dygraph import layers
from paddle.fluid.layers.utils import flatten
from paddle.fluid.layers.utils import pack_sequence_as
from paddle.fluid.framework import convert_np_dtype_to_dtype_, Variable
from paddle.fluid.dygraph.base import switch_to_static_graph
from paddle.fluid.dygraph.dygraph_to_static.utils import parse_arg_and_kwargs
from paddle.fluid.dygraph.dygraph_to_static.utils import type_name


class TensorSpec(object):

    __slots__ = ['_shape', '_dtype', '_name']

    def __init__(self, shape, dtype='float32', name=None):
        # replace `None` in shape  with -1
        self._shape = self._verify(shape)
        # convert dtype into united represention
        if dtype is not None:
            if not isinstance(dtype, core.VarDesc.VarType):
                dtype = convert_np_dtype_to_dtype_(dtype)
        self._dtype = dtype
        self._name = name

    @classmethod
    def from_variable(cls, variable, name=None):
        if isinstance(variable, (Variable, core.VarBase)):
            return cls(variable.shape, variable.dtype, name or variable.name)
        else:
            raise ValueError(
                "Input `variable` should be a Variable, but received {}.".
                format(type_name(variable)))

    @classmethod
    def from_numpy(cls, ndarray, name=None):
        return cls(ndarray.shape, ndarray.dtype, name)

    def batch(self, batch_size):
        """
        Insert `batch_size` in front of the `shape`.
        """
        if isinstance(batch_size, (list, tuple)):
            if len(batch_size) != 1:
                raise ValueError(
                    "Length of  {}: batch_size shall be 1, but received {}.".
                    format(type_name(variable), len(batch_size)))
            batch_size = batch_size[1]
        elif not isinstance(batch_size, (int, long)):
            raise TypeError(
                "type(batch_size) shall be int or long, but received {}.".
                format(type_name(batch_size)))

        new_shape = [batch_size] + list(self._shape)
        return TensorSpec(new_shape, self._dtype, self._name)

    def unbatch(self):
        if len(self._shape) == 0:
            raise ValueError(
                "Not support to unbatch a variable when len(shape) == 0.")

        return TensorSpec(self._shape[1:], self._dtype, self.name)

    def _verify(self, shape):
        if not isinstance(shape, (list, tuple)):
            raise TypeError(
                "Type of `shape` in TensorSpec should be one of (tuple, list), but received {}.".
                format(type_name(shape)))
        if len(shape) == 0:
            raise ValueError(
                "`shape` in TensorSpec should contains at least 1 element, but received empty shape."
            )

        for i, ele in enumerate(shape):
            if ele is not None:
                if not isinstance(ele, (int, long)):
                    raise ValueError(
                        "shape[{}] should be a int, but received `{}`:{}.".
                        format(i, type_name(ele), ele))
            if ele is None or ele < -1:
                shape[i] = -1

        return tuple(shape)

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return '{}(shape={}, dtype={}, name={})'.format(
            type_name(self), self._shape, self._dtype, self._name)

    def __hash__(self):
        return hash((self._shape, self._dtype))

    def __eq__(self, other):
        return (type(self) is type(other) and all(
            getattr(self, attr) == getattr(other, attr)
            for attr in self.__slots__))

    def __ne__(self, other):
        return not self == other


def get_parameters(layer_instance, include_sublayer=True):
    """
    Returns parameters of decorated layers. If set `include_sublayer` True,
    the parameters created in sub layers will be added.
    """
    params = collections.OrderedDict()
    if layer_instance is not None:
        if isinstance(layer_instance, layers.Layer):
            if include_sublayer:
                params = layer_instance.parameters()
                names = [p.name for p in params]
                params = collections.OrderedDict(zip(names, params))
            else:
                params = layer_instance._parameters
        else:
            raise TypeError(
                "Type of `layer_instance` should be layer.Layer, but received {}".
                format(type_name(layer_instance)))

    return params


def get_buffers(layer_instance, include_sublayer=True):
    """
    Returns Variable buffers of decorated layers. If set `include_sublayer` True,
    the Variable buffers created in sub layers will be added.
    """
    buffers = collections.OrderedDict()
    if layer_instance is not None:
        if isinstance(layer_instance, layers.Layer):
            if include_sublayer:
                buffers = layer_instance.buffers()
                names = [buffer.name for buffer in buffers]
                buffers = collections.OrderedDict(zip(names, buffers))
            else:
                buffers = layer_instance._buffers
        else:
            raise TypeError(
                "Type of `layer_instance` should be layer.Layer, but received {}".
                format(type_name(layer_instance)))
    return buffers


def convert_inputs_to_tensor_spec(inputs, inputs_spec):
    """
    Replace tensor in structured `inputs` by tensorSpec in `inputs_spec`.
    """

    def check_type_and_len(input, spec, check_length=False):
        if type(input) is not type(spec):
            raise TypeError('type(inputs) should be {}, but received {}.'.
                            format(type(spec), type(input)))
        if check_length and len(input) < len(spec):
            raise ValueError(
                'Requires len(inputs) >= len(inputs_spec), but received len(inputs):{} < len(inputs_spec):{}'.
                format(len(inputs), len(inputs_spec)))

    if isinstance(inputs_spec, (tuple, list)):
        inputs_with_spec = []
        check_type_and_len(inputs, inputs_spec, True)

        for i, spec in enumerate(inputs_spec):
            out_spec = convert_inputs_to_tensor_spec(inputs[i], spec)
            inputs_with_spec.append(out_spec)

        # TODO(Aurelius84): If the rest inputs contains tensor or numpy.ndarray,
        # should we enhance check and raise warning for users?
        inputs_with_spec.extend(inputs[len(inputs_spec):])
        return inputs_with_spec
    elif isinstance(inputs_spec, dict):
        inputs_with_spec = {}
        check_type_and_len(inputs, inputs_spec, True)
        for name, input in inputs.items():
            if name in inputs_spec:
                inputs_with_spec[name] = convert_inputs_to_tensor_spec(
                    input, inputs_spec[name])
            else:
                print('{} has no spec info.'.format(name))
                inputs_with_spec[name] = input
        return inputs_with_spec
    elif isinstance(inputs_spec, TensorSpec):
        return inputs_spec
    else:
        raise TypeError(
            "type(inputs_spec) should be `TensorSpec` or dict/list/tuple of it, but received {}.".
            type_name(inputs_spec))


class FunctionSpec(object):
    def __init__(self, function, input_spec=None, is_method=False):
        self._dyfunc = function
        if input_spec is None:
            self._input_spec = None
            self._flatten_spec = None
        else:
            if not isinstance(input_spec, (tuple, list)):
                raise TypeError(
                    "Type of `input_spec` should be one of (tuple, list), but received {}.".
                    format(type_name(input_spec)))
            self._input_spec = tuple(input_spec)
            self._flatten_signature = flatten(self._input_spec)

        self._is_method = is_method
        # parse full argument names list.
        self._arg_names, self._default_kwargs = parse_arg_and_kwargs(function)

    def unified_args_and_kwargs(self, args, kwargs):
        """
        Move kwarg with default value into arguments list.
        
        For example: 
        
        Given function definition: `def foo(x, a=1, b=2)`, 
        when calling it by `foo(23)`, the args is `[23]`, kwargs is `{a=1, b=2}`.
        In this function, it will return args with `[23, 1, 2]`, kwargs with `{}`
        """
        if len(self._arg_names) < len(args):
            error_msg = "The decorated function `{}` requires {} arguments: {}, but received {} with {}.".format(
                self._dyfunc.__name__,
                len(self._arg_names), self._arg_names, len(args), args)
            if args and inspect.isclass(args[0]):
                error_msg += "\n\tMaybe the function has more than one decorator, we don't support this for now."
                raise NotImplementedError(error_msg)
            else:
                raise ValueError(error_msg)

        args = list(args)
        for i in six.moves.range(len(args), len(self._arg_names)):
            arg_name = self._arg_names[i]
            if arg_name in kwargs:
                args.append(kwargs[arg_name])
                del kwargs[arg_name]
            else:
                args.append(self._default_kwargs[arg_name])

        return tuple(args), kwargs

    def args_to_variable_spec(self, args, kwargs):
        """
        Convert input arguments into TensorSpec.
        
        1. If specific input_spec, use them to construct feed layers.
        2. If input_spec is None, consider all Tensor and Numpy.ndarray as feed layers
        """
        inputs_with_spec = []

        if self._input_spec is not None:
            if kwargs:
                raise ValueError(
                    "{} got unexpected keyword arguments: {}. Cannot trace the function when input_spec is specificed.".
                    format(self._dyfunc.__name__, kwargs))

            if len(args) < len(self._input_spec):
                raise ValueError(
                    "Requires len(arguments) >= len(input_spec), but received len(args):{} < len(TensorSpec): {}".
                    format(len(args), len(self._input_spec)))

            inputs_with_spec = convert_inputs_to_tensor_spec(args,
                                                             self._input_spec)
        else:
            for idx, input_var in enumerate(flatten(args)):
                if isinstance(input_var, np.ndarray):
                    input_var = TensorSpec.from_numpy(input_var)
                elif isinstance(input_var, core.VarBase):
                    input_var = TensorSpec.from_variable(input_var)

                if isinstance(input_var, TensorSpec):
                    self._idx_to_variable_spec[idx] = input_var
                inputs_with_spec.append(input_var)

            inputs_with_spec = pack_sequence_as(args, inputs_with_spec)

        return inputs_with_spec

    @switch_to_static_graph
    def to_static_inputs_with_spec(self, inputs_with_spec, main_program):
        flat_inputs_spec = flatten(inputs_with_spec)

        inputs = []
        block = main_program.global_block()
        for i, var_spec in enumerate(flat_inputs_spec):
            if isinstance(var_spec, TensorSpec):
                feed_layer = block.create_var(
                    # TODO: consider more elegant way to name this
                    name=var_spec.name or "feed_%s" % i,
                    shape=var_spec.shape,
                    dtype=var_spec.dtype,
                    is_data=True,
                    need_check_feed=False)
            else:
                feed_layer = var_spec
            inputs.append(feed_layer)

        return pack_sequence_as(inputs_with_spec, inputs)

    @property
    def dyfunc(self):
        return self._dyfunc

    @property
    def args(self):
        return self._args

    @property
    def is_method(self):
        return self._is_method

    @property
    def code(self):
        dyfunc = getattr(self._function, '__wrapped__', self._function)
        return inspect.getsource(dyfunc)
