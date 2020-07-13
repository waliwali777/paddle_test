#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: define random functions  

import numpy as np

from ..fluid import core
from ..fluid.framework import device_guard, in_dygraph_mode, _varbase_creator, Variable, convert_np_dtype_to_dtype_
from ..fluid.layers.layer_function_generator import templatedoc
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype
from ..fluid.layers import uniform_random, utils
from ..fluid.layers.tensor import fill_constant

from ..fluid.io import shuffle  #DEFINE_ALIAS

__all__ = [
    #       'gaussin',
    #       'uniform',
    'shuffle',
    'randn',
    'rand',
    'randint',
    'randperm'
]


def randint(low=0, high=None, shape=[1], dtype=None, name=None):
    """
	:alias_main: paddle.randint
	:alias: paddle.randint,paddle.tensor.randint,paddle.tensor.random.randint

    This function returns a Tensor filled with random integers from the
    "discrete uniform" distribution of the specified data type in the interval
    [low, high). If high is None (the default), then results are from [0, low).

    Args:
        low (int): The lower bound on the range of random values to generate,
            the low is included in the range.(unless high=None, in which case
            this parameter is one above the highest such integer). Default is 0.
        high (int, optional): The upper bound on the range of random values to
            generate, the high is excluded in the range. Default is None(see
            above for behavior if high=None).
        shape (list|tuple|Variable, optional): The shape of the output Tensor,
            if the shape is a list or tuple, its elements can be an integer or
            a Tensor with the shape [1], and the type of the Tensor must be
            int32 or int64. If the shape is a Variable, it is a 1-D Tensor,
            and the type of the Tensor must be int32 or int64. Default is None.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): Data type of the
            output Tensor which can be int32, int64. If dtype is `None`, the
            data type of created Tensor is `int64`
        name(str, optional): The default value is None.  Normally there is no
            need for user to set this property.  For more information, please
            refer to :ref:`api_guide_Name`.

    Returns: 
        Variable: A Tensor of the specified shape filled with random integers.

    Raises:
        ValueError: Randint's low must less then high.
        TypeError: shape's type must be list, tuple or Variable.
        TypeError: dtype must be int32 or int64.

    Examples:
        .. code-block:: python

        import paddle
        import numpy as np

        paddle.enable_imperative()

        # example 1:
        # attr shape is a list which doesn't contain tensor Variable.
        result_1 = paddle.randint(low=-5, high=5, shape=[3])
        # [0 -3 2]

        # example 2:
        # attr shape is a list which contains tensor Variable.
        dim_1 = paddle.fill_constant([1],"int64",2)
        dim_2 = paddle.fill_constant([1],"int32",3)
        result_2 = paddle.randint(low=-5, high=5, shape=[dim_1, dim_2], dtype="int32")
        print(result_2.numpy())
        # [[ 0 -1 -3]
        #  [ 4 -2  0]]

        # example 3:
        # attr shape is a Variable
        var_shape = paddle.imperative.to_variable(np.array([3]))
        result_3 = paddle.randint(low=-5, high=5, shape=var_shape)
        # [-2 2 3]

        # example 4:
        # date type is int32
        result_4 = paddle.randint(low=-5, high=5, shape=[3], dtype='int32')
        # [-5 4 -4]

        # example 5:
        # Input only one parameter
        # low=0, high=10, shape=[1], dtype='int64'
        result_5 = paddle.randint(10)
        # [7]

    """
    if high is None:
        high = low
        low = 0
    if dtype is None:
        dtype = 'int64'
    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dygraph_mode():
        shape = utils._convert_shape_to_list(shape)
        return core.ops.randint('shape', shape, 'low', low, 'high', high,
                                'seed', 0, 'dtype', dtype)

    check_type(shape, 'shape', (list, tuple, Variable), 'randint')
    check_dtype(dtype, 'dtype', ['int32', 'int64'], 'randint')
    if low >= high:
        raise ValueError(
            "randint's low must less then high, but received low = {0}, "
            "high = {1}".format(low, high))

    inputs = dict()
    attrs = {'low': low, 'high': high, 'seed': 0, 'dtype': dtype}
    utils._get_shape_tensor_inputs(
        inputs=inputs, attrs=attrs, shape=shape, op_type='randint')

    helper = LayerHelper("randint", **locals())
    out = helper.create_variable_for_type_inference(dtype=dtype)
    helper.append_op(
        type='randint', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    return out


def randn(shape,
          out=None,
          dtype=None,
          device=None,
          stop_gradient=True,
          name=None):
    """
	:alias_main: paddle.randn
	:alias: paddle.randn,paddle.tensor.randn,paddle.tensor.random.randn

    This function returns a tensor filled with random numbers from a normal 
    distribution with mean 0 and variance 1 (also called the standard normal
    distribution).

    Args:
        shape(list|tuple): Shape of the generated random tensor.
        out(Variable, optional): Optional output which can be any created Variable 
            that meets the requirements to store the result of operation. If the 
            out is `None`, a new Variable will be returned to store the result.
            Default is None.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): Data type of the output 
            tensor, which can be float32, float64. if dtype is `None` , the data 
            type of output tensor is `float32` .
            Default is None.
        device(str, optional): Specific the output variable to be saved in cpu
            or gpu memory. Supported None, 'cpu', 'gpu'. If it is None, the output
            variable will be automatically assigned devices. 
            Default: None.
        stop_gradient(bool, optional): Indicating if we stop gradient from current(out) 
            Variable. Default is True.
        name(str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .
            Default is None.

    Returns:
        Random tensor whose data is drawn from a standard normal distribution,
        dtype: flaot32 or float64 as specified.

    Return type:
        Variable

    Raises:
        TypeError: If the type of `shape` is not list or tuple.
        TypeError: If the data type of `dtype` is not float32 or float64.
        ValueError: If the length of `shape` is not bigger than 0.

    Examples:
        .. code-block:: python

            # declarative mode
            import paddle
            import paddle.fluid as fluid

            data = paddle.randn([2, 4])
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            res, = exe.run(fluid.default_main_program(), feed={}, fetch_list=[data])
            print(res)
            # [[-1.4187592   0.7368311  -0.53748125 -0.0146909 ]
            #  [-0.66294265 -1.3090698   0.1898754  -0.14065823]]

        .. code-block:: python

            # imperative mode
            import paddle
            import paddle.fluid as fluid
            import paddle.fluid.dygraph as dg

            place = fluid.CPUPlace()
            with dg.guard(place) as g:
                x = paddle.randn([2, 4])
                x_np = x.numpy()
                print(x_np)
                # [[ 1.5149173  -0.26234224 -0.592486    1.4523455 ]
                #  [ 0.04581212 -0.85345626  1.1687907  -0.02512913]]
    """
    helper = LayerHelper("randn", **locals())
    check_type(shape, 'shape', (list, tuple), 'randn')
    assert len(shape) > 0, ("The size of argument(shape) can't be zero.")

    if dtype is None:
        dtype = 'float32'

    check_dtype(dtype, 'create data type', ['float32', 'float64'], 'randn')

    if out is None:
        out = helper.create_variable_for_type_inference(dtype=dtype)
    else:
        check_variable_and_dtype(out, 'out', [dtype], 'randn')

    out.stop_gradient = stop_gradient

    dtype = convert_np_dtype_to_dtype_(dtype)
    seed = np.random.randint(0, 100)

    with device_guard(device):
        helper.append_op(
            type='gaussian_random',
            outputs={'Out': out},
            attrs={
                'shape': shape,
                'mean': 0.0,
                'std': 1.0,
                'seed': seed,
                'dtype': dtype,
                'use_mkldnn': False
            })
    return out


@templatedoc()
def randperm(n, dtype="int64", name=None):
    """
	:alias_main: paddle.randperm
	:alias: paddle.randperm,paddle.tensor.randperm,paddle.tensor.random.randperm

    ${comment}

    Args:
        n(int): The upper bound (exclusive), and it should be greater than 0.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): The type of the 
            output Tensor. Supported data types: int32, int64, float32, float64.
            Default: int32.
        name(str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .
            Default is None.

    Returns:
        ${out_comment}.

    Return Type:
        ${out_type}

    Examples:
        .. code-block:: python

        import paddle

        paddle.enable_imperative()

        result_1 = paddle.randperm(5)
        # [4 1 2 3 0]

        result_2 = paddle.randperm(7, 'int32')
        # [1 6 2 0 4 3 5]
 
    """
    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dygraph_mode():
        return core.ops.randperm('n', n, 'seed', 0, 'dtype', dtype)

    if n < 1:
        raise ValueError("The input n should be greater than 0 in randperm op.")
    check_dtype(dtype, 'dtype', ['int64', 'int32', 'float32', 'float64'],
                'randperm')

    helper = LayerHelper("randperm", **locals())
    out = helper.create_variable_for_type_inference(dtype)
    attrs = {'n': n, 'dtype': dtype, 'seed': 0}
    helper.append_op(
        type='randperm', inputs={}, outputs={'Out': out}, attrs=attrs)
    return out


def rand(shape, dtype=None, name=None):
    """
	:alias_main: paddle.rand
	:alias: paddle.rand,paddle.tensor.rand,paddle.tensor.random.rand

    This OP initializes a variable with random values sampled from a
    uniform distribution in the range [0, 1).

    Examples:
    ::

        Input:
          shape = [1, 2]

        Output:
          result=[[0.8505902, 0.8397286]]

    Args:
        shape(list|tuple|Variable): Shape of the Tensor to be created. The data
            type is ``int32`` or ``int64`` . If ``shape`` is a list or tuple,
            the elements of it should be integers or Tensors with shape [1]. If
            ``shape`` is a Variable, it should be an 1-D Tensor .
        dtype(np.dtype|core.VarDesc.VarType|str, optional): Data type of the
            output tensor which can be float32, float64, if dytpe is `None`,
            the data type of created tensor is `float32`
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.
    Returns:
        Variable: A Tensor of the specified shape filled with random numbers
        from a uniform distribution on the interval [0, 1).

    Raises:
        TypeError: The shape type should be list or tupple or Variable.

    Examples:
        .. code-block:: python

        import paddle
        import numpy as np

        paddle.enable_imperative()
        # example 1: attr shape is a list which doesn't contain tensor Variable.
        result_1 = paddle.rand(shape=[2, 3])
        # [[0.451152  , 0.55825245, 0.403311  ],
        #  [0.22550228, 0.22106001, 0.7877319 ]]

        # example 2: attr shape is a list which contains tensor Variable.
        dim_1 = paddle.fill_constant([1], "int64", 2)
        dim_2 = paddle.fill_constant([1], "int32", 3)
        result_2 = paddle.rand(shape=[dim_1, dim_2, 2])
        # [[[0.8879919  0.25788337]
        #   [0.28826773 0.9712097 ]
        #   [0.26438272 0.01796806]]
        #  [[0.33633623 0.28654453]
        #   [0.79109055 0.7305809 ]
        #   [0.870881   0.2984597 ]]]

        # example 3: attr shape is a Variable, the data type must be int64 or int32.
        var_shape = paddle.imperative.to_variable(np.array([2, 3]))
        result_3 = paddle.rand(var_shape)
        # [[0.22920267 0.841956   0.05981819]
        #  [0.4836288  0.24573246 0.7516129 ]]

    """
    if dtype is None:
        dtype = 'float32'
    return uniform_random(shape, dtype, min=0.0, max=1.0, name=name)
