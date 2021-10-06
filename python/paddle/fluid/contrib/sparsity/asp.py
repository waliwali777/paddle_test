# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
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
"""
Functions for Auto SParsity (ASP) training and inference.
"""

import copy
import numpy as np
import paddle
from paddle.fluid import framework, global_scope, program_guard, layers
from paddle.fluid.initializer import ConstantInitializer
from paddle.fluid.contrib import sparsity
from paddle.fluid import core

__all__ = [
    'decorate', 'prune_model', 'set_excluded_layers', 'reset_excluded_layers'
]


def set_excluded_layers(main_program, param_names):
    r"""
    Set parameter name of layers which would not be pruned as sparse weights.

    Args:
        main_program (Program, optional): Program with model definition and its parameters.
        param_names (list): A list contains names of parameters.
    """
    ASPHelper.set_excluded_layers(
        main_program=main_program, param_names=param_names)


def reset_excluded_layers(main_program=None):
    r"""
    Reset exculded layers setting corresponding to :attr:`main_program`. If :attr:`main_program` 
    is None, then all configurations of excluded_layers would be cleaned.

    Args:
        main_program (Program, optional): Program with model definition and its parameters.
    """
    ASPHelper.reset_excluded_layers(main_program=main_program)


def add_supported_layer(layer, pruning_func=None):
    r"""
    Add supported layers and its corresponding pruning functino.

    Args:
        name (string|Layer): The name or type of layer, needed to support. If layer is `Layer` then 
        it would be turn to string internally. ASP would use this name to match parameter's name and call 
        its the corresponding pruning function.
        pruning_func (function, optional): a function type which receives five argument (weight_nparray,
        m, n, func_name, param_name), weight_nparray is a nparray of weight, param_name is the name of weight,
        m, n, and func_name, please see `prune_model` for details.
    """
    name = None
    if isinstance(layer, str):
        name = layer
    elif issubclass(layer, paddle.fluid.dygraph.layers.Layer):
        name = paddle.fluid.dygraph.layers._convert_camel_to_snake(
            layer.__name__)
    else:
        assert "The type of layer should be string of Layer, but got {}!".format(
            type(layer))
    ASPHelper.add_supported_layer(name, pruning_func)


def decorate(optimizer):
    r"""
    Wrap the given optimizer as a OptimizerWithSparsityGuarantee, 
    which would insert necessary ops for ASP workflows when calling minimize()

    Args:
        optimizer (Optimizer): A Optimizer used for training.
    Returns:
        OptimizerWithSparsityGuarantee: A wrapper for ASP to decorate `minimize` function of the given optimizer.
    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            from paddle.fluid.contrib import sparsity

            main_program = fluid.Program()
            startup_program = fluid.Program()

            paddle.enable_static()

            with fluid.program_guard(main_program, startup_program):
                input_data = fluid.layers.data(name='data', shape=[None, 128])
                label = fluid.layers.data(name='label', shape=[None, 10])
                hidden = fluid.layers.fc(input=input_data, num_flatten_dims=-1, size=32, act=None)
                prob = fluid.layers.fc(input=hidden, num_flatten_dims=-1, size=10, act=None)
                loss = fluid.layers.mean(fluid.layers.square_error_cost(prob, label))

                optimizer = fluid.optimizer.SGD(learning_rate=0.1)
                optimizer = sparsity.decorate(optimizer)
                # if do sparse training with Fleet, please replace above decorate with:
                # strategy = paddle.distributed.fleet.DistributedStrategy()
                # strategy.asp = True
                # optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)

                optimizer.minimize(loss, startup_program)
    """
    return ASPHelper.decorate(optimizer)


def prune_model(place,
                main_program=None,
                n=2,
                m=4,
                func_name=sparsity.MaskAlgo.MASK_1D,
                with_mask=True):
    r"""
    Pruning parameters of supported layers in :attr:`main_program` via 
    specified mask generation function given by :attr:`func_name`. This 
    function supports both training and inference controlled by :attr:`with_mask`.
    If :attr:`with_mask` is True, it would also prune parameter related ASP mask Variables,
    else only prunes parameters.

    *Note*: If parameters are supported and in FP16, please set :attr:`n`=2, :attr:`m`=4, 
    if they in FP32, then :attr:`n`=1, :attr:`m`=2` to further enable Sparse Tensor Core acceleration.

    *Note*: If calling this function with :attr:`with_mask`, it should call `OptimizerWithSparsityGuarantee.minimize` 
    and initialization (`exe.run(startup_program`)) before (For successfully obtain mask Variable). 
    Typically set `with_mask` as true for training (have called `OptimizerWithSparsityGuarantee.minimize`) and false for 
    inference only. To obtain OptimizerWithSparsityGuarantee, please see `sparsity.decoreate()`.

    Args:
        place (fluid.CPUPlace()|fluid.CUDAPlace(N)): Device place for pruned parameter and mask Variables, and N means the GPU's id. It should be the same as created instance of Executor.
        main_program (Program, optional): Program with model definition and its parameters. Default is `paddle.static.default_main_program()
        n (int): n of `n:m` sparse pattern.
        m (int): m of `n:m` sparse pattern.
        func_name (MaskAlgo, optional): The function name to generate spase mask. Default is `MaskAlgo.MASK_1D`. All options please refer to `MaskAlgo`.
        with_mask (bool, optional): To prune mask Variables related to parameters or not. Ture is purning also, False is not. Defalut is True.
    Returns:
        dictionary: A dictionary with key: `parameter name` (string) and value: its corresponding mask Variable.
    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import paddle.fluid.core as core
            from paddle.fluid.contrib import sparsity

            paddle.enable_static()

            main_program = fluid.Program()
            startup_program = fluid.Program()

            place = paddle.CPUPlace()
            if core.is_compiled_with_cuda():
                place = paddle.CUDAPlace(0)

            with fluid.program_guard(main_program, startup_program):
                input_data = fluid.layers.data(name='data', shape=[None, 128])
                label = fluid.layers.data(name='label', shape=[None, 10])
                hidden = fluid.layers.fc(input=input_data, num_flatten_dims=-1, size=32, act=None, name="need_sparse")
                hidden = fluid.layers.fc(input=hidden, num_flatten_dims=-1, size=32, act=None, name="need_dense")
                prob = fluid.layers.fc(input=hidden, num_flatten_dims=-1, size=10, act=None)
                loss = fluid.layers.mean(fluid.layers.square_error_cost(prob, label))

                # Setup exluded layers out from ASP workflow.
                # Please note, excluded_layers must be set before calling `optimizer.minimize()`.
                sparsity.set_excluded_layers(main_program, ["need_dense"])

                optimizer = fluid.optimizer.SGD(learning_rate=0.1)
                optimizer = fluid.contrib.mixed_precision.decorator.decorate(optimizer )
                # Calling sparsity.decorate() to wrap minimize() in optimizer, which 
                # will insert necessary masking operations for ASP workflow.
                optimizer = sparsity.decorate(optimizer)
                optimizer.minimize(loss, startup_program)

            exe = fluid.Executor(place)
            exe.run(startup_program)

            # Must call `exe.run(startup_program)` first before calling `sparsity.prune_model`
            sparsity.prune_model(place, main_program, func_name=sparsity.MaskAlgo.MASK_2D_BEST)
    """
    return ASPHelper.prune_model(
        place=place,
        main_program=main_program,
        n=n,
        m=m,
        func_name=func_name,
        with_mask=with_mask)


class ProgramASPInfo(object):
    r"""
    ProgramASPInfo is a container to keep ASP relevant information of Pragrom. It contains three inner-variables:
    1. __mask_vars (Dictionary): Key is parameter's name and vaule is its corresponding sparse mask Variable object, which is created by `ASPHelper.create_mask_variables`.
    2. __masks (Dictionary): Key is parameter's name and vaule is its corressponding sparse mask Numpy array, which is created by `ASPHelper.prune_model`.
    3. __excluded_layers (List): It stores name of layers which should not involve into ASP workflow.
    """

    def __init__(self):
        self.__mask_vars = {}
        self.__masks = {}
        self.__excluded_layers = []

    def update_mask_vars(self, param_name, var):
        self.__mask_vars[param_name] = var

    def update_masks(self, param_name, var):
        self.__masks[param_name] = var

    def update_excluded_layers(self, param_names):
        self.__excluded_layers.extend(copy.deepcopy(param_names))

    def reset_excluded_layers(self):
        self.__excluded_layers = []

    @property
    def mask_vars(self):
        return self.__mask_vars

    @property
    def masks(self):
        return self.__masks

    @property
    def excluded_layers(self):
        return self.__excluded_layers


class ASPHelper(object):
    r"""
    ASPHelper is a collection of Auto SParsity (ASP) functions to enable 

    1. training models with weights in 2:4 sparse pattern on FP16 or 1:2 sparse pattern on FP32 from scratch.
    2. pruning well-trained models into 2:4 sparse pattern on FP16 or 1:2 sparse pattern on FP32 for fine-tuning.
    """

    MASK_APPENDDED_NAME = '_asp_mask'
    PADDLE_WEIGHT_SUFFIX = "w_"
    # When value of given key in this DICT is None, 
    # ASP will call default pruning function in pruning stage.
    SUPPORTED_LAYERS = set(['fc', 'linear', 'conv'])
    LAYERS_AND_PRUNE_FUNC_MAP = {}

    __asp_info = {}

    @classmethod
    def set_excluded_layers(cls, main_program, param_names):
        r"""
        This is the implementation of `sparsity.set_excluded_layers`, for details please see explanation in `sparsity.set_excluded_layers`.
        """
        asp_info = cls._get_program_asp_info(main_program)
        asp_info.update_excluded_layers(param_names)

    @classmethod
    def reset_excluded_layers(cls, main_program=None):
        r"""
        This is the implementation of `sparsity.reset_excluded_layers`, for details please see explanation in `sparsity.reset_excluded_layers`.
        """
        if main_program is None:
            for asp_info in cls.__asp_info:
                asp_info.reset_excluded_layers()
        else:
            cls._get_program_asp_info(main_program).reset_excluded_layers()

    @classmethod
    def add_supported_layer(cls, name, pruning_func=None):
        cls.SUPPORTED_LAYERS.add(name)
        if pruning_func is not None:
            cls.LAYERS_AND_PRUNE_FUNC_MAP.update({name: pruning_func})

    @staticmethod
    def decorate(optimizer):
        r"""
        This is the implementation of `sparsity.decorate`, for details please see explanation in `sparsity.decorate`.
        """
        return OptimizerWithSparsityGuarantee(optimizer)

    @classmethod
    def prune_model(cls,
                    place,
                    main_program=None,
                    n=2,
                    m=4,
                    func_name=sparsity.MaskAlgo.MASK_1D,
                    with_mask=True):
        r"""
        This is the implementation of `sparsity.prune_model`, for details please see explanation in `sparsity.prune_model`.
        """

        if main_program is None:
            main_program = paddle.static.default_main_program()

        asp_info = cls._get_program_asp_info(main_program)
        for param in main_program.global_block().all_parameters():
            if ASPHelper._is_supported_layer(main_program, param.name):
                weight_tensor = global_scope().find_var(param.name).get_tensor()
                weight_nparray = np.array(weight_tensor)

                prune_func = None
                for layer_name in ASPHelper.LAYERS_AND_PRUNE_FUNC_MAP:
                    if layer_name in param.name:
                        prune_func = ASPHelper.LAYERS_AND_PRUNE_FUNC_MAP[
                            layer_name]
                        break
                if prune_func is None:
                    weight_pruned_nparray, weight_sparse_mask = \
                        ASPHelper._default_pruning(weight_nparray, m, n, func_name, param.name)
                else:
                    weight_pruned_nparray, weight_sparse_mask = \
                        prune_func(weight_nparray, m, n, func_name, param.name)
                weight_tensor.set(weight_pruned_nparray, place)
                # weight_sparse_mask = sparsity.create_mask(
                #     weight_nparray.T, func_name=func_name, n=n, m=m).T
                # weight_pruned_nparray = np.multiply(weight_nparray,
                #                                     weight_sparse_mask)
                # weight_tensor.set(weight_pruned_nparray, place)
                # assert sparsity.check_sparsity(weight_pruned_nparray.T,  n=n, m=m, func_name=checked_func_name), \
                #         'Pruning {} weight matrix failure!!!'.format(param.name)
                if with_mask:
                    weight_mask_param = global_scope().find_var(
                        ASPHelper._get_mask_name(param.name))
                    assert weight_mask_param is not None, \
                        'Cannot find {} variable, please call ASPHelper.minimize' \
                        ' and initialization (exe.run(startup_program)) first!'.format(ASPHelper._get_mask_name(param.name))
                    weight_mask_tensor = weight_mask_param.get_tensor()
                    weight_mask_tensor.set(weight_sparse_mask, place)
                asp_info.update_masks(param.name, weight_sparse_mask)
        return asp_info.masks.copy()

    @staticmethod
    def _default_pruning(weight_nparray, m, n, func_name, param_name):

        checked_func_name = sparsity.CheckMethod.get_checking_method(func_name)

        # The double transpose ops here make sure pruning direction consistent with cuSparseLt.
        # SPMMA in cuSparseLt: D = (AxB) + C, where matrix A (mxk) is sparse matrix.
        # cuSparseLt would prune matrix A along k dimension.
        # In sparse training, layer weight matriices is viewed sparse matrix A, so
        # the math fomula should be 'Act(WX + b)'. However, default fomula in PaddlePaddle
        #  is 'Act(XW + b)'. For enabling SPMMA, weights and inputs should be transposed 
        # for computing, Act( (W^T X^T)^T + b). Therefore, we have to prune alog k dimension 
        # of W^T, which is m dimension of W. Moreove, all mask generating functions in 
        # sparsity/utils is row-major pruning. That is the reason we have to transpose weight 
        # matrices beforce invoking create_mask. Then we transpose the result maks to make 
        # sure its shape to be the same as the input weight.
        weight_sparse_mask = sparsity.create_mask(
            weight_nparray.T, func_name=func_name, n=n, m=m).T
        weight_pruned_nparray = np.multiply(weight_nparray, weight_sparse_mask)
        assert sparsity.check_sparsity(weight_pruned_nparray.T,  n=n, m=m, func_name=checked_func_name), \
                        'Pruning {} weight matrix failure!!!'.format(param_name)
        return weight_pruned_nparray, weight_sparse_mask

    @staticmethod
    def _get_mask_name(param_name):
        r"""
        Return mask name by given parameter name :attr:`param_name`.

        Args:
            param_name (string): The name of parameter.
        Returns:
            string: The mask name of :attr:`param_name`.
        """
        return param_name + ASPHelper.MASK_APPENDDED_NAME

    @staticmethod
    def _get_not_ASP_relevant_vars(main_program):
        r"""
        Get all parameters's Variables in :attr:`main_program` but excluded ASP mask Variables.

        Args:
            main_program (Program): Program with model definition and its parameters.
        Returns:
            list: A list of parameter Variables in :attr:`main_program` (excluded ASP mask Variables).
        """
        var_list = []
        for param in main_program.global_block().all_parameters():
            if ASPHelper.MASK_APPENDDED_NAME not in param.name:
                var_list.append(param)
        return var_list

    @classmethod
    def _get_program_asp_info(cls, main_program):
        if not main_program in cls.__asp_info:
            cls.__asp_info[main_program] = ProgramASPInfo()
        return cls.__asp_info[main_program]

    @classmethod
    def _is_supported_layer(cls, main_program, param_name):
        r"""
        Verify if given :attr:`param_name` is supported by ASP.

        Args:
            param_name (string): The name of parameter.
        Returns:
            bool: True if it is supported, else False.
        Examples:
            .. code-block:: python

              import paddle.fluid as fluid
              from paddle.fluid.contrib.sparsity.asp import ASPHelper

              main_program = fluid.Program()
              startup_program = fluid.Program()

              with fluid.program_guard(main_program, startup_program):
                  input_data = fluid.layers.data(name='data', shape=[None, 128])
                  fc = fluid.layers.fc(input=input_data, num_flatten_dims=-1, size=32, act=None)

              for param in main_program.global_block().all_parameters():
                  ASPHelper._is_supported_layer(main_program, param.name)
              # fc_0.w_0 -> True
              # fc_0.b_0 -> False
        """
        if ASPHelper.MASK_APPENDDED_NAME in param_name:
            return False

        for layer in cls._get_program_asp_info(main_program).excluded_layers:
            if layer in param_name:
                return False

        for name in ASPHelper.SUPPORTED_LAYERS:
            if name in param_name and \
               ASPHelper.PADDLE_WEIGHT_SUFFIX in param_name:
                return True
        return False

    @classmethod
    def _minimize(cls,
                  optimizer,
                  loss,
                  main_program=None,
                  startup_program=None,
                  parameter_list=None,
                  no_grad_set=None):
        r"""
        This function is a decorator of `minimize` function in `Optimizer`.
        There are three steps:

        1. Call :attr:`optimizer`.minimize(:attr:`loss`)
        2. Create sparse mask Tensors according to supported layers in :attr:`main_program`.
        3. Insert masking ops in the end of parameters update.

        *Note*: Please use `ASP.decorate` instead when applying distributed training with `Fleet`. 
        (Due to there is a invisiable graphs optimization in `Fleet.minimize()` which make training graph 
        cannot be modified anymore.)

        Args:
            optimizer (Optimizer): A Optimizer used for training.
            loss (Variable): A Variable containing the value to minimize.
            main_program (Program, optional): Program with model definition and its parameters. Default is `loss.block.program`.
            startup_program (Program, optional): Program for initializing parameters in `parameter_list`. Default is `paddle.static.default_startup_program()`.
            parameter_list (Iterable, optional): Iterable of `Variable` or `Variable.name` to update to minimize `loss`. The default value is None, at this time all parameters will be updated.
            no_grad_set (set, optional): Set of `Variable  or `Variable.name` that don't need to be updated. The default value is None.
        Returns:
            list: operators from :attr:`optimizer`.minimize(:attr:`loss`).
            list: pairs of parameters and their gradients.
        """
        if main_program is None:
            main_program = loss.block.program

        if startup_program is None:
            startup_program = paddle.static.default_startup_program()

        optimizer_ops, params_and_grads = optimizer.minimize(
            loss, startup_program, parameter_list, no_grad_set=no_grad_set)
        cls._create_mask_variables(main_program, startup_program,
                                   params_and_grads)
        cls._insert_sparse_mask_ops(main_program, params_and_grads)
        return optimizer_ops, params_and_grads

    @classmethod
    def _create_mask_variables(cls, main_program, startup_program,
                               params_and_grads):
        r"""
        Create sparse mask Tensors according to supported layers in :attr:`main_program`.
        This function is called in second step of `ASPHelper._minimize`

        Args:
            main_program (Program): Program with model definition and its parameters.
            startup_program (Program): Program for initializing parameters.
            params_and_grads (list): Variable pairs of parameters and their gradients.
        """
        asp_info = cls._get_program_asp_info(main_program)
        with program_guard(main_program, startup_program):
            for param_and_grad in params_and_grads:
                if ASPHelper._is_supported_layer(main_program,
                                                 param_and_grad[0].name):
                    mask_param = layers.create_parameter(
                        name=param_and_grad[0].name +
                        ASPHelper.MASK_APPENDDED_NAME,
                        shape=param_and_grad[0].shape,
                        dtype=param_and_grad[0].dtype,
                        default_initializer=ConstantInitializer(value=1.0))
                    mask_param.stop_gradient = True
                    mask_param.trainable = False
                    asp_info.update_mask_vars(param_and_grad[0].name,
                                              mask_param)

    @classmethod
    def _insert_sparse_mask_ops(cls, main_program, param_grads):
        r"""
        Insert masking ops in the end of parameters update.
        This function is called in third step of `ASPHelper._minimize`

        Args:
            main_program (Program): Program with model definition and its parameters.
            params_and_grads (list): Variable pairs of parameters and their gradients.
        """
        block = main_program.global_block()
        asp_info = cls._get_program_asp_info(main_program)
        for param_grad in param_grads:
            if param_grad[0].name in asp_info.mask_vars:
                block.append_op(
                    type='elementwise_mul',
                    inputs={
                        "X": param_grad[0],
                        'Y': asp_info.mask_vars[param_grad[0].name]
                    },
                    outputs={'Out': param_grad[0]},
                    attrs={'axis': -1,
                           'use_mkldnn': False})


class OptimizerWithSparsityGuarantee(object):
    r"""
    OptimizerWithSparsityGuarantee is a wrapper to decorate `minimize` function of given optimizer by `_minimize` of ASPHelper.
    The decorated `minimize` function would do three things (exactly same as `ASPHelper._minimize`):
    1. Call `minimize` function of given optimizer.
    2. Call `ASPHelper._create_mask_variables` to create mask Variables.
    3. Call `ASPHelper._insert_sparse_mask_ops` to insert weight masking ops in the end of `loss`'s Program.
    """

    def __init__(self, optimizer):
        self._optimizer = optimizer
        self._learning_rate = optimizer._learning_rate
        self._learning_rate_map = optimizer._learning_rate_map

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        r"""
        This function is to call `ASPHelper.minimize()` and return its return

        Args:
            loss (Variable): A Variable containing the value to minimize.
            startup_program (Program, optional): Program for initializing parameters in `parameter_list`. Default is `paddle.static.default_startup_program()`.
            parameter_list (Iterable, optional): Iterable of `Variable` or `Variable.name` to update to minimize `loss`. The default value is None, at this time all parameters will be updated.
            no_grad_set (set, optional): Set of `Variable  or `Variable.name` that don't need to be updated. The default value is None.
        Returns:
            list: operators from :attr:`optimizer`.minimize(:attr:`loss`).
            list: pairs of parameters and their gradients.
        """
        return ASPHelper._minimize(
            self._optimizer,
            loss,
            startup_program=startup_program,
            parameter_list=parameter_list,
            no_grad_set=no_grad_set)
