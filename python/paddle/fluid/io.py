#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import os
import errno
import warnings
import six
import logging
import pickle
import contextlib
from functools import reduce

import numpy as np

import paddle
import paddle.reader
from paddle.reader import *
from paddle.fluid import layers
from paddle.fluid.executor import Executor, global_scope
from paddle.fluid.evaluator import Evaluator
from paddle.fluid.framework import Program, Parameter, default_main_program, default_startup_program, Variable, \
    program_guard
from paddle.fluid.compiler import CompiledProgram
from paddle.fluid.log_helper import get_logger
from . import reader
from .reader import *
from . import core
from .. import compat as cpt

batch = paddle.batch

__all__ = [
    'save_vars',
    'save_params',
    'save_persistables',
    'load_vars',
    'load_params',
    'load_persistables',
    'save_inference_model',
    'load_inference_model',
    'batch',
    'save',
    'load',
    'load_program_state',
    'set_program_state',
    'get_program_parameter',
    'get_program_persistable_vars',
] + reader.__all__ + paddle.reader.__all__

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


def is_parameter(var):
    """
    Check whether the given variable is an instance of Parameter.

    Args:
        var(Variable): The variable to be checked.

    Returns:
        bool: True if the given `var` is an instance of Parameter,
        False if not.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            param = fluid.default_main_program().global_block().var('fc.w')
            res = fluid.io.is_parameter(param)
    """
    return isinstance(var, Parameter)


def is_persistable(var):
    """
    Check whether the given variable is persistable.

    Args:
        var(Variable): The variable to be checked.

    Returns:
        bool: True if the given `var` is persistable
        False if not.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            param = fluid.default_main_program().global_block().var('fc.b')
            res = fluid.io.is_persistable(param)
    """
    if var.desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or \
            var.desc.type() == core.VarDesc.VarType.FETCH_LIST or \
            var.desc.type() == core.VarDesc.VarType.READER:
        return False
    return var.persistable


def is_belong_to_optimizer(var):
    if not (isinstance(var, Parameter) or var.desc.need_check_feed()):
        return is_persistable(var)

    return False


def get_program_parameter(program):
    """
    Get all the parameters from Program.

    Args:
        var(Program): The Program to get parameters

    Returns:
        list: The list contains all parameters in the program

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            data = fluid.data(name="img", shape=[64, 784])
            w = fluid.layers.create_parameter(shape=[784, 200], dtype='float32', name='fc_w')
            b = fluid.layers.create_parameter(shape=[200], dtype='float32', name='fc_b')
            list_para  = fluid.io.get_program_parameter(  fluid.default_main_program() )
    """
    return list(filter(is_parameter, program.list_vars()))


def get_program_persistable_vars(program):
    """
    Get all the persistable vars from Program.

    Args:
        var(Program): The Program to get persistable vars

    Returns:
        list: The list contains all persistable vars in the program

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            data = fluid.data(name="img", shape=[64, 784])
            w = fluid.layers.create_parameter(shape=[784, 200], dtype='float32', name='fc_w')
            b = fluid.layers.create_parameter(shape=[200], dtype='float32', name='fc_b')
            list_para  = fluid.io.get_program_persistable_vars(  fluid.default_main_program() )
    """
    return list(filter(is_persistable, program.list_vars()))


def _clone_var_in_block_(block, var):
    assert isinstance(var, Variable)
    if var.desc.type() == core.VarDesc.VarType.LOD_TENSOR:
        return block.create_var(
            name=var.name,
            shape=var.shape,
            dtype=var.dtype,
            type=var.type,
            lod_level=var.lod_level,
            persistable=True)
    else:
        return block.create_var(
            name=var.name,
            shape=var.shape,
            dtype=var.dtype,
            type=var.type,
            persistable=True)


@contextlib.contextmanager
def _load_program_scope(main=None, startup=None, scope=None):
    prog = main if main else paddle.fluid.Program()
    startup_prog = startup if startup else paddle.fluid.Program()
    scope = scope if scope else paddle.fluid.core.Scope()
    with paddle.fluid.scope_guard(scope):
        with paddle.fluid.program_guard(prog, startup_prog):
            with paddle.fluid.unique_name.guard():
                yield


def _get_valid_program(main_program):
    if main_program is None:
        main_program = default_main_program()
    elif isinstance(main_program, CompiledProgram):
        main_program = main_program._program
        if main_program is None:
            raise TypeError("program should be as Program type or None")
        warnings.warn(
            "The input is a CompiledProgram, this is not recommended.")
    if not isinstance(main_program, Program):
        raise TypeError("program should be as Program type or None")
    return main_program


def save_vars(executor,
              dirname,
              main_program=None,
              vars=None,
              predicate=None,
              filename=None):
    """
    This API saves specific variables in the `Program` to files.

    There are two ways to specify the variables to be saved: set variables in 
    a list and assign it to the `vars`, or use the `predicate` function to select
    variables that make `predicate(variable) == True`. The first way has a higher priority.

    The `dirname` is used to specify the folder where to save variables.
    If you prefer to save variables in separate files in the `dirname` folder,
    do not set `filename`. If you prefer to save all variables in a single file,
    use `filename` to specify it.

    Args:
        executor(Executor): The executor to run for saving variables.
        dirname(str): The folder where to save variables.
        main_program(Program, optional): The program whose variables will be saved.
                                    If it is None, the default main program will
                                    be used automatically.
                                    Default: None
        vars(list[Variable], optional): The list contains all variables to be saved.
                                        Default: None
        predicate(function, optional): The function selects the variables that make
                                       `predicate(variable) == True`. 
                                       Default: None
        filename(str, optional): If you prefer to save all variables in a single file,
                                 use `filename` to specify it. Otherwise, let `filename` be None. 
                                 Default: None

    Returns:
        None

    Raises:
        TypeError: If `main_program` is not an instance of Program nor None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            main_prog = fluid.Program()
            startup_prog = fluid.Program()
            with fluid.program_guard(main_prog, startup_prog):
                data = fluid.layers.data(name="img", shape=[64, 784], append_batch_size=False)
                w = fluid.layers.create_parameter(shape=[784, 200], dtype='float32', name='fc_w')
                b = fluid.layers.create_parameter(shape=[200], dtype='float32', name='fc_b')
                hidden_w = fluid.layers.matmul(x=data, y=w)
                hidden_b = fluid.layers.elementwise_add(hidden_w, b)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_prog)

            # The first usage: use `vars` to set the saved variables.
            var_list = [w, b]
            path = "./my_paddle_vars"
            fluid.io.save_vars(executor=exe, dirname=path, vars=var_list,
                            filename="vars_file")
            # w and b will be save in a file named "var_file".

            # The second usage: use `predicate` to select the saved variable.
            def name_has_fc(var):
                res = "fc" in var.name
                return res
            param_path = "./my_paddle_model"
            fluid.io.save_vars(executor=exe, dirname=param_path, main_program=main_prog, vars=None, predicate = name_has_fc)
            # all variables whose names contain "fc " are saved.
    """
    save_dirname = os.path.normpath(dirname)
    main_program = _get_valid_program(main_program)

    if vars is None:
        save_vars(
            executor,
            main_program=main_program,
            dirname=save_dirname,
            vars=list(filter(predicate, main_program.list_vars())),
            filename=filename)
    else:
        # give warning when there is no var in model
        if len(list(vars)) == 0:
            warnings.warn(
                "no variable in your model, please ensure there are any variables in your model to save"
            )
            return None

        save_program = Program()
        save_block = save_program.global_block()

        save_var_map = {}
        for each_var in vars:
            # NOTE: don't save the variable which type is RAW
            if each_var.type == core.VarDesc.VarType.RAW:
                continue
            new_var = _clone_var_in_block_(save_block, each_var)
            if filename is None:
                save_file_path = os.path.join(save_dirname, new_var.name)
                save_file_path = os.path.normpath(save_file_path)
                save_block.append_op(
                    type='save',
                    inputs={'X': [new_var]},
                    outputs={},
                    attrs={'file_path': save_file_path})
            else:
                save_var_map[new_var.name] = new_var

        if filename is not None:
            save_var_list = []
            for name in sorted(save_var_map.keys()):
                save_var_list.append(save_var_map[name])

            save_block.append_op(
                type='save_combine',
                inputs={'X': save_var_list},
                outputs={},
                attrs={'file_path': os.path.join(save_dirname, filename)})

        #NOTE(zhiqiu): save op will add variable kLookupTablePath in save_program.desc,
        # which leads to diff on save_program and its desc. Call _sync_with_cpp
        # to keep consistency.
        save_program._sync_with_cpp()
        executor.run(save_program)


def save_params(executor, dirname, main_program=None, filename=None):
    """
    This operator saves all parameters from the :code:`main_program` to
    the folder :code:`dirname` or file :code:`filename`. You can refer to 
    :ref:`api_guide_model_save_reader_en` for more details.

    Use the :code:`dirname` to specify the saving folder. If you would like to
    save parameters in separate files, set :code:`filename` None; if you would
    like to save all parameters in a single file, use :code:`filename` to specify
    the file name.

    Note: 
        Some variables are not Parameter while they are necessary for
        training, such as learning rate, global step, etc. So you can NOT save 
        and continue your training just by :ref:`api_fluid_io_save_params`
        and :ref:`api_fluid_io_load_params`. Please use :ref:`api_fluid_io_save_persistables`
        and :ref:`api_fluid_io_load_persistables` instead. 
        
        If you want to save your model for the inference, please use the 
        :ref:`api_fluid_io_save_inference_model`. You can refer to
        :ref:`api_guide_model_save_reader_en` for more details.

    Args:
        executor(Executor): The executor to run for saving parameters, You can 
                            refer to :ref:`api_guide_executor_en`.
        dirname(str): The saving directory path.
        main_program(Program, optional): The program whose parameters will be
                                         saved. You can refer to 
                                         :ref:`api_guide_Program_en` for more 
                                         details. If it is None, the default main
                                         program will be used.
                                         Default: None
        filename(str, optional): The file to save all parameters. If you prefer
                                 to save parameters in different files, set it
                                 to None.
                                 Default: None

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
           
            params_path = "./my_paddle_model"
            image = fluid.data(name='img', shape=[None, 28, 28], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            feeder = fluid.DataFeeder(feed_list=[image, label], place=fluid.CPUPlace())
            predict = fluid.layers.fc(input=image, size=10, act='softmax')
    
            loss = fluid.layers.cross_entropy(input=predict, label=label)
            avg_loss = fluid.layers.mean(loss)
            
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())
            fluid.io.save_params(executor=exe, dirname=params_path)
            # The parameters weights and bias of the fc layer in the network are going to 
            # be saved in different files in the path "./my_paddle_model" 
    """
    save_vars(
        executor,
        dirname=dirname,
        main_program=main_program,
        vars=None,
        predicate=is_parameter,
        filename=filename)


def _save_distributed_persistables(executor, dirname, main_program):
    """
    save_persistables for distributed training.
    the method will do things listed below:
    1.save part of persistable variables on trainer.
    2.receive "remote prefetch variables" from parameter servers and merge them.
    3.save "distributed lookup table" on parameter servers.
    4.receive "optimizer variables" from parameter servers and merge them.

    Args:
        executor(Executor): The executor to run for saving parameters.
        dirname(str): The saving directory path.
        main_program(Program): The program whose parameters will be
                            saved. the main_program must be the trainer_program
                            get after transpiler.

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            exe = fluid.Executor(fluid.CPUPlace())
            param_path = "./my_paddle_model"
            t = distribute_transpiler.DistributeTranspiler()
            t.transpile(...)
            train_program = t.get_trainer_program()
            _save_distributed_persistables(executor=exe, dirname=param_path, main_program=train_program)
    """

    def __save_remote_params(executor, dirname, remote_params_map):
        """
        receive params on pserver through rpc.
        if the params are be sliced, will concat them to one, then save it.
        """
        if not remote_params_map:
            return

        prog = Program()
        block = prog.global_block()

        # recv optimize vars from pserver
        for name, remote_params in remote_params_map.items():
            origin = remote_params[0].origin
            is_slice = remote_params[0].is_slice

            slices = [None] * len(remote_params)
            slice_varnames = [None] * len(remote_params)
            remote_varnames = [None] * len(remote_params)
            endpoints = [None] * len(remote_params)

            for idx, optimizer in enumerate(remote_params):
                block_id = optimizer.block_id
                slice = optimizer.slice
                endpoint = optimizer.endpoint

                index = block_id if is_slice else idx
                slices[index] = slice
                slice_varnames[index] = "{}.slice.{}".format(slice.name, idx)
                remote_varnames[index] = slice.name
                endpoints[index] = endpoint

            slice_shapes = []
            for slice in slices:
                tmp = [str(dim) for dim in slice.shape]
                slice_shapes.append(",".join(tmp))

            block.append_op(
                type='recv_save',
                attrs={
                    "trainer_id": 0,
                    "shape": origin.shape,
                    "slice_shapes": slice_shapes,
                    "slice_varnames": slice_varnames,
                    "remote_varnames": remote_varnames,
                    "endpoints": endpoints,
                    "file_path": os.path.join(dirname, origin.name)
                })

        executor.run(prog)

    def __save_distributed_lookup_tables(executor, dirname,
                                         distributed_lookup_table, endpoints):
        """
        because the distributed lookup table may too huge to merge and save at one place,
        it will be saved at parameter server independent respectively.

        the save directory is dirname/"__lookup_table__".

        """
        prog = Program()
        block = prog.global_block()

        # if there is lookup table, the trainer 0 will notify all pserver to save.
        lookup_table_filename = os.path.join(dirname, "__lookup_table__")
        attrs = {}
        attrs['epmap'] = endpoints
        attrs['dir'] = lookup_table_filename
        attrs['lookup_table'] = distributed_lookup_table
        block.append_op(
            type='checkpoint_notify', inputs={}, outputs={}, attrs=attrs)
        executor.run(prog)

    def __exclude_vars(exclude_var_names=[]):
        def is_valid(var):
            if var.name in exclude_var_names:
                return False
            if var.desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or \
                    var.desc.type() == core.VarDesc.VarType.FETCH_LIST or \
                    var.desc.type() == core.VarDesc.VarType.READER:
                return False
            return var.persistable

        return is_valid

    if not isinstance(main_program, Program):
        raise TypeError("'main_program' should be an instance of Program.")

    if not main_program._is_distributed:
        raise ValueError(
            "'_save_distributed_persistables' just be designed for distributed training."
        )

    remote_params_map = main_program._parameters_on_pservers.get_distributed_vars_by_vtypes(
        ["Optimizer", "RemotePrefetch"], groupby=True)

    exclude_var_names = []
    if remote_params_map:
        exclude_var_names.extend(remote_params_map.keys())

    if main_program._distributed_lookup_table:
        if isinstance(main_program._distributed_lookup_table, list):
            exclude_var_names.extend(main_program._distributed_lookup_table)
        else:
            exclude_var_names.append(main_program._distributed_lookup_table)

    local_vars = list(
        filter(__exclude_vars(exclude_var_names), main_program.list_vars()))
    save_vars(
        executor, main_program=main_program, dirname=dirname, vars=local_vars)

    if main_program._is_chief:
        if remote_params_map:
            __save_remote_params(executor, dirname, remote_params_map)
        if main_program._distributed_lookup_table:
            __save_distributed_lookup_tables(
                executor, dirname, main_program._distributed_lookup_table,
                main_program._endpoints)


def save_persistables(executor, dirname, main_program=None, filename=None):
    """
    This operator saves all persistable variables from :code:`main_program` to 
    the folder :code:`dirname` or file :code:`filename`. You can refer to 
    :ref:`api_guide_model_save_reader_en` for more details. And then
    saves these persistables variables to the folder :code:`dirname` or file 
    :code:`filename`. 

    The :code:`dirname` is used to specify the folder where persistable variables
    are going to be saved. If you would like to save variables in separate
    files, set :code:`filename` None; if you would like to save all variables in a
    single file, use :code:`filename` to specify the file name.

    Args:
        executor(Executor): The executor to run for saving persistable variables.
                            You can refer to :ref:`api_guide_executor_en` for 
                            more details.
        dirname(str): The saving directory path.
        main_program(Program, optional): The program whose persistable variables will
                                         be saved. You can refer to 
                                         :ref:`api_guide_Program_en` for more details.
                                         If it is None, the default main program will 
                                         be used.
                                         Default: None.
        filename(str, optional): The file to save all variables. If you prefer to
                                 save variables in different files, set it to None.
                                 Default: None.

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
        
            dir_path = "./my_paddle_model"
            file_name = "persistables"
            image = fluid.data(name='img', shape=[None, 28, 28], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            feeder = fluid.DataFeeder(feed_list=[image, label], place=fluid.CPUPlace())
           
            predict = fluid.layers.fc(input=image, size=10, act='softmax')
            loss = fluid.layers.cross_entropy(input=predict, label=label)
            avg_loss = fluid.layers.mean(loss)
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())
            fluid.io.save_persistables(executor=exe, dirname=dir_path, filename=file_name)
            # The persistables variables weights and bias in the fc layer of the network 
            # are going to be saved in the same file named "persistables" in the path
            # "./my_paddle_model"
    """
    if main_program and main_program._is_distributed:
        _save_distributed_persistables(
            executor, dirname=dirname, main_program=main_program)
    else:
        save_vars(
            executor,
            dirname=dirname,
            main_program=main_program,
            vars=None,
            predicate=is_persistable,
            filename=filename)


def load_vars(executor,
              dirname,
              main_program=None,
              vars=None,
              predicate=None,
              filename=None):
    """
    This API loads variables from files by executor.

    There are two ways to specify the variables to be loaded: the first way, set
    variables in a list and assign it to the `vars`; the second way, use the 
    `predicate` function to select variables that make `predicate(variable) == True`. 
    The first way has a higher priority.

    The `dirname` is used to specify the folder where to load variables.
    If variables were saved in separate files in the folder `dirname`,
    set `filename` None. If all variables were saved in a single file,
    use `filename` to specify it.

    Args:
        executor(Executor): The executor to run for loading variables.
        dirname(str): The folder where to load the variables.
        main_program(Program, optional): The program whose variables will be loaded.
                                    If it is None, the default main program will
                                    be used automatically.
                                    Default: None
        vars(list[Variable], optional): The list that contains all variables to be loaded.
                                   Default: None
        predicate(function, optional): The function selects variables that make 
                                        `predicate(variable) == True`.
                                        Default: None
        filename(str, optional): The file which saved all required variables. If variables
                                were saved in separate files, set it to be None.
                                Default: None

    Returns:
        None

    Raises:
        TypeError: If `main_program` is not an instance of Program nor None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            main_prog = fluid.Program()
            startup_prog = fluid.Program()
            with fluid.program_guard(main_prog, startup_prog):
                data = fluid.layers.data(name="img", shape=[64, 784], append_batch_size=False)
                w = fluid.layers.create_parameter(shape=[784, 200], dtype='float32', name='fc_w')
                b = fluid.layers.create_parameter(shape=[200], dtype='float32', name='fc_b')
                hidden_w = fluid.layers.matmul(x=data, y=w)
                hidden_b = fluid.layers.elementwise_add(hidden_w, b)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_prog)

            # The first usage: using `vars` to specify the variables.
            path = "./my_paddle_vars"
            var_list = [w, b]
            fluid.io.save_vars(executor=exe, dirname=path, vars=var_list,
                               filename="vars_file")
            fluid.io.load_vars(executor=exe, dirname=path, vars=var_list,
                               filename="vars_file")
            # w and b will be loaded, and they are supposed to
            # be saved in the same file named 'var_file' in the path "./my_paddle_vars".

            # The second usage: using the `predicate` function to select variables
            param_path = "./my_paddle_model"
            def name_has_fc(var):
                res = "fc" in var.name
                return res
            fluid.io.save_vars(executor=exe, dirname=param_path, main_program=main_prog,
                              vars=None, predicate=name_has_fc)
            fluid.io.load_vars(executor=exe, dirname=param_path, main_program=main_prog,
                               vars=None, predicate=name_has_fc)
            # Load All variables in the `main_program` whose name includes "fc".
            # And all the variables are supposed to be saved in separate files.

    """
    load_dirname = os.path.normpath(dirname)

    if vars is None:
        if main_program is None:
            main_program = default_main_program()
        if not isinstance(main_program, Program):
            raise TypeError("program's type should be Program")

        load_vars(
            executor,
            dirname=load_dirname,
            main_program=main_program,
            vars=list(filter(predicate, main_program.list_vars())),
            filename=filename)
    else:
        load_prog = Program()
        load_block = load_prog.global_block()

        if main_program is None:
            main_program = default_main_program()

        if not isinstance(main_program, Program):
            raise TypeError("program should be as Program type or None")

        # save origin param shape
        orig_para_shape = {}
        load_var_map = {}
        for each_var in vars:
            assert isinstance(each_var, Variable)
            if each_var.type == core.VarDesc.VarType.RAW:
                continue

            if isinstance(each_var, Parameter):
                orig_para_shape[each_var.name] = tuple(each_var.desc.get_shape(
                ))
            new_var = _clone_var_in_block_(load_block, each_var)
            if filename is None:
                load_block.append_op(
                    type='load',
                    inputs={},
                    outputs={'Out': [new_var]},
                    attrs={
                        'file_path': os.path.join(load_dirname, new_var.name)
                    })
            else:
                load_var_map[new_var.name] = new_var

        if filename is not None:
            load_var_list = []
            for name in sorted(load_var_map.keys()):
                load_var_list.append(load_var_map[name])

            load_block.append_op(
                type='load_combine',
                inputs={},
                outputs={"Out": load_var_list},
                attrs={'file_path': os.path.join(load_dirname, filename)})
        executor.run(load_prog)

        # check var shape
        for each_var in vars:
            if not isinstance(each_var, Parameter):
                continue
            var_temp = paddle.fluid.global_scope().find_var(each_var.name)
            assert var_temp != None, "can't not find var: " + each_var.name
            new_shape = (np.array(var_temp.get_tensor())).shape
            assert each_var.name in orig_para_shape, each_var.name + "MUST in var list"
            orig_shape = orig_para_shape.get(each_var.name)
            if new_shape != orig_shape:
                raise RuntimeError(
                    "Shape not matching: the Program requires a parameter with a shape of ({}), "
                    "while the loaded parameter (namely [ {} ]) has a shape of  ({}).".
                    format(orig_shape, each_var.name, new_shape))


def load_params(executor, dirname, main_program=None, filename=None):
    """
    This API filters out all parameters from the give ``main_program``
    and then tries to load these parameters from the directory ``dirname`` or
    the file ``filename``.

    Use the ``dirname`` to specify the directory where parameters were saved. If
    parameters were saved in separate files under the directory `dirname`, set
    ``filename`` as None; if all parameters were saved in a single file, use
    ``filename`` to specify the file name.

    **Note**:
        Some variables are not Parameter while they are necessary for
        training, such as learning rate, global step, etc. So you cannot save and
        continue your training just by using :ref:`api_fluid_io_save_params` and
        :ref:`api_fluid_io_load_params`. Please use :ref:`api_fluid_io_save_persistables`
        and :ref:`api_fluid_io_load_persistables` instead.

        If you want to load the pre-trained model structure and parameters
        for the inference, please use the :ref:`api_fluid_io_load_inference_model` API. You can
        refer to :ref:`api_guide_model_save_reader_en` for more details.

    Args:
        executor(Executor): The executor used for loading parameters.
                            See :ref:`api_guide_executor_en` for more details about it.
        dirname(str): The directory path.
        main_program(Program, optional): The program whose parameters will be
                                    loaded. If it is None, the ``default_main_program``
                                    will be used automatically. See :ref:`api_guide_Program_en`
                                    for more about ``Program``.
                                    Default: None.
        filename(str, optional): The file which saved all parameters. If parameters
                            were saved in separated files, set it to None.
                            Default: None.

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            exe = fluid.Executor(fluid.CPUPlace())
            param_path = "./my_paddle_model"
            prog = fluid.default_main_program()
            fluid.io.load_params(executor=exe, dirname=param_path,
                                main_program=None)
    """
    load_vars(
        executor,
        dirname=dirname,
        main_program=main_program,
        predicate=is_parameter,
        filename=filename)


def load_persistables(executor, dirname, main_program=None, filename=None):
    """
    This API filters out all variables with ``persistable==True`` from the
    given ``main_program`` and then tries to load these variables from the
    directory ``dirname`` or the file ``filename``.

    Use the ``dirname`` to specify the directory where persistable variables
    (refer to :ref:`api_guide_model_save_reader_en`) were saved. If variables
    were saved in separate files, set ``filename`` as None; if all variables
    were saved in a single file, use ``filename`` to specify the file name.

    Args:
        executor(Executor): The executor used for loading persistable variables.
                            See :ref:`api_guide_executor_en` for more details about it.
        dirname(str): The directory path.
        main_program(Program, optional): The program whose persistable variables will
                                    be loaded. If it is None, the ``default_main_program``
                                    will be used automatically. See :ref:`api_guide_Program_en`
                                    for more about ``Program``.
                                    Default: None.
        filename(str, optional): The file which saved all persistable variables. If variables
                                 were saved in separated files, set it to None.
                                 Default: None.

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            exe = fluid.Executor(fluid.CPUPlace())
            param_path = "./my_paddle_model"
            prog = fluid.default_main_program()
            fluid.io.load_persistables(executor=exe, dirname=param_path,
                                       main_program=None)
    """

    if main_program and main_program._is_distributed:
        _load_distributed_persistables(
            executor, dirname=dirname, main_program=main_program)
    else:
        load_vars(
            executor,
            dirname=dirname,
            main_program=main_program,
            predicate=is_persistable,
            filename=filename)


def _load_distributed_persistables(executor, dirname, main_program=None):
    """
    customized load_persistables for distributed training.
    it should be used on parameter server,

    Args:
        executor(Executor): The executor to run for saving parameters.
        dirname(str): The load directory path.
        main_program(Program): The program whose parameters will be
                            loaded. the main_program must be the pserver_program
                            get after transpiler.

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            exe = fluid.Executor(fluid.CPUPlace())
            param_path = "./my_paddle_model"
            t = distribute_transpiler.DistributeTranspiler()
            t.transpile(...)
            pserver_prog = t.get_pserver_program(...)
            _load_distributed_persistables(executor=exe, dirname=param_path, main_program=pserver_prog)
    """

    def __is_distributed_part_var(varname):
        trainer_idx = varname.find(".trainer_")
        block_idx = varname.find(".block")
        return trainer_idx or block_idx

    def __load_persistable_vars(executor, dirname, need_load_vars):
        load_prog = Program()
        load_block = load_prog.global_block()
        need_delete_vars = []

        for param in need_load_vars:
            origin_var = param.origin
            slice_var = param.slice
            is_slice = param.is_slice
            offset = param.offset

            if is_slice:
                slice = load_block.create_var(
                    name=slice_var.name,
                    type=slice_var.type,
                    shape=slice_var.shape,
                    dtype=slice_var.dtype,
                    persistable=True)

                load_block.append_op(
                    type='load',
                    inputs={},
                    outputs={'Out': [slice]},
                    attrs={
                        'file_path': os.path.join(dirname, origin_var.name),
                        'seek': offset,
                        'shape': slice.shape
                    })
            else:
                origin = load_block.create_var(
                    name="{}".format(origin_var.name),
                    type=origin_var.type,
                    shape=origin_var.shape,
                    dtype=origin_var.dtype,
                    persistable=True)
                load_block.append_op(
                    type='load',
                    inputs={},
                    outputs={'Out': [origin]},
                    attrs={
                        'file_path': os.path.join(dirname, origin_var.name)
                    })

        load_block.append_op(
            type='delete_var',
            inputs={'X': need_delete_vars}, )

        executor.run(load_prog)

    if not isinstance(main_program, Program):
        raise TypeError("'main_program' should be an instance of Program.")

    if not main_program._is_distributed:
        raise ValueError(
            "'_load_distributed_persistables' just be designed for distributed training."
        )

    if not main_program._ps_endpoint:
        raise ValueError(
            "'_load_distributed_persistables' need current_endpoint set in DistributeTranspiler.transpile"
        )

    need_load_vars = main_program._parameters_on_pservers.get_distributed_vars_by_ep(
        main_program._ps_endpoint)
    __load_persistable_vars(executor, dirname, need_load_vars)


def prepend_feed_ops(inference_program,
                     feed_target_names,
                     feed_holder_name='feed'):
    if len(feed_target_names) == 0:
        return

    global_block = inference_program.global_block()
    feed_var = global_block.create_var(
        name=feed_holder_name,
        type=core.VarDesc.VarType.FEED_MINIBATCH,
        persistable=True)

    for i, name in enumerate(feed_target_names):
        out = global_block.var(name)
        global_block._prepend_op(
            type='feed',
            inputs={'X': [feed_var]},
            outputs={'Out': [out]},
            attrs={'col': i})


def append_fetch_ops(inference_program,
                     fetch_target_names,
                     fetch_holder_name='fetch'):
    global_block = inference_program.global_block()
    fetch_var = global_block.create_var(
        name=fetch_holder_name,
        type=core.VarDesc.VarType.FETCH_LIST,
        persistable=True)

    for i, name in enumerate(fetch_target_names):
        global_block.append_op(
            type='fetch',
            inputs={'X': [name]},
            outputs={'Out': [fetch_var]},
            attrs={'col': i})


def save_inference_model(dirname,
                         feeded_var_names,
                         target_vars,
                         executor,
                         main_program=None,
                         model_filename=None,
                         params_filename=None,
                         export_for_deployment=True,
                         program_only=False):
    """
    Prune the given `main_program` to build a new program especially for inference,
    and then save it and all related parameters to given `dirname` .
    If you just want to save parameters of your trained model, please use the
    :ref:`api_fluid_io_save_params` . You can refer to :ref:`api_guide_model_save_reader_en`
    for more details.

    Note:
        The :code:`dirname` is used to specify the folder where inference model 
        structure and parameters are going to be saved. If you would like to save params of
        Program in separate files, set `params_filename` None; if you would like to save all 
        params of Program in a single file, use `params_filename` to specify the file name.

    Args:
        dirname(str): The directory path to save the inference model.
        feeded_var_names(list[str]): list of string. Names of variables that need to be fed
                                     data during inference.
        target_vars(list[Variable]): list of Variable. Variables from which we can get 
                                     inference results.
        executor(Executor): The executor that saves the inference model. You can refer 
                            to :ref:`api_guide_executor_en` for more details.
        main_program(Program, optional): The original program, which will be pruned to
                                         build the inference model. If is set None,
                                         the global default :code:`_main_program_` will be used.
                                         Default: None.
        model_filename(str, optional): The name of file to save the inference program
                                       itself. If is set None, a default filename
                                       :code:`__model__` will be used.
        params_filename(str, optional): The name of file to save all related parameters.
                                        If it is set None, parameters will be saved
                                        in separate files .
        export_for_deployment(bool): If True, programs are modified to only support
                                     direct inference deployment. Otherwise,
                                     more information will be stored for flexible
                                     optimization and re-training. Currently, only
                                     True is supported.
                                     Default: True.
        program_only(bool, optional): If True, It will save inference program only, and do not 
                                      save params of Program.
                                      Default: False.

    Returns:
        The fetch variables' name list

     Return Type:
        list

    Raises:
        ValueError: If `feed_var_names` is not a list of basestring, an exception is thrown.
        ValueError: If `target_vars` is not a list of Variable, an exception is thrown.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            path = "./infer_model"

            # User defined network, here a softmax regession example
            image = fluid.data(name='img', shape=[None, 28, 28], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            feeder = fluid.DataFeeder(feed_list=[image, label], place=fluid.CPUPlace())
            predict = fluid.layers.fc(input=image, size=10, act='softmax')

            loss = fluid.layers.cross_entropy(input=predict, label=label)
            avg_loss = fluid.layers.mean(loss)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            # Feed data and train process

            # Save inference model. Note we don't save label and loss in this example
            fluid.io.save_inference_model(dirname=path,
                                          feeded_var_names=['img'],
                                          target_vars=[predict],
                                          executor=exe)

            # In this example, the save_inference_mode inference will prune the default
            # main program according to the network's input node (img) and output node(predict). 
            # The pruned inference program is going to be saved in the "./infer_model/__model__"
            # and parameters are going to be saved in separate files under folder
            # "./infer_model".

    """
    if isinstance(feeded_var_names, six.string_types):
        feeded_var_names = [feeded_var_names]
    elif export_for_deployment:
        if len(feeded_var_names) > 0:
            # TODO(paddle-dev): polish these code blocks
            if not (bool(feeded_var_names) and all(
                    isinstance(name, six.string_types)
                    for name in feeded_var_names)):
                raise ValueError("'feed_var_names' should be a list of str.")

    if isinstance(target_vars, Variable):
        target_vars = [target_vars]
    elif export_for_deployment:
        if not (bool(target_vars) and
                all(isinstance(var, Variable) for var in target_vars)):
            raise ValueError("'target_vars' should be a list of Variable.")

    main_program = _get_valid_program(main_program)

    # remind user to set auc_states to zeros if the program contains auc op 
    all_ops = main_program.global_block().ops
    for op in all_ops:
        # clear device of Op
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
        op._set_attr(device_attr_name, "")
        if op.type == 'auc':
            warnings.warn(
                "please ensure that you have set the auc states to zeros before saving inference model"
            )
            break

    # fix the bug that the activation op's output as target will be pruned.
    # will affect the inference performance.
    # TODO(Superjomn) add an IR pass to remove 1-scale op.
    with program_guard(main_program):
        uniq_target_vars = []
        for i, var in enumerate(target_vars):
            if isinstance(var, Variable):
                var = layers.scale(
                    var, 1., name="save_infer_model/scale_{}".format(i))
            uniq_target_vars.append(var)
        target_vars = uniq_target_vars
    target_var_name_list = [var.name for var in target_vars]

    # when a pserver and a trainer running on the same machine, mkdir may conflict
    save_dirname = dirname
    try:
        save_dirname = os.path.normpath(dirname)
        os.makedirs(save_dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    if model_filename is not None:
        model_basename = os.path.basename(model_filename)
    else:
        model_basename = "__model__"
    model_basename = os.path.join(save_dirname, model_basename)

    # When export_for_deployment is true, we modify the program online so that
    # it can only be loaded for inference directly. If it's false, the whole
    # original program and related meta are saved so that future usage can be
    # more flexible.

    origin_program = main_program.clone()

    if export_for_deployment:
        main_program = main_program.clone()
        global_block = main_program.global_block()
        need_to_remove_op_index = []
        for i, op in enumerate(global_block.ops):
            op.desc.set_is_target(False)
            if op.type == "feed" or op.type == "fetch":
                need_to_remove_op_index.append(i)

        for index in need_to_remove_op_index[::-1]:
            global_block._remove_op(index)

        main_program.desc.flush()

        main_program = main_program._prune_with_input(
            feeded_var_names=feeded_var_names, targets=target_vars)
        main_program = main_program._inference_optimize(prune_read_op=True)
        fetch_var_names = [v.name for v in target_vars]

        prepend_feed_ops(main_program, feeded_var_names)
        append_fetch_ops(main_program, fetch_var_names)

        main_program.desc._set_version()
        paddle.fluid.core.save_op_compatible_info(main_program.desc)
        with open(model_basename, "wb") as f:
            f.write(main_program.desc.serialize_to_string())
    else:
        # TODO(panyx0718): Save more information so that it can also be used
        # for training and more flexible post-processing.
        with open(model_basename + ".main_program", "wb") as f:
            f.write(main_program.desc.serialize_to_string())

    if program_only:
        warnings.warn(
            "save_inference_model specified the param `program_only` to True, It will not save params of Program."
        )
        return target_var_name_list

    main_program._copy_dist_param_info_from(origin_program)

    if params_filename is not None:
        params_filename = os.path.basename(params_filename)

    save_persistables(executor, save_dirname, main_program, params_filename)
    return target_var_name_list


def load_inference_model(dirname,
                         executor,
                         model_filename=None,
                         params_filename=None,
                         pserver_endpoints=None):
    """
    Load the inference model from a given directory. By this API, you can get the model
    structure(Inference Program) and model parameters. If you just want to load
    parameters of the pre-trained model, please use the :ref:`api_fluid_io_load_params` API.
    You can refer to :ref:`api_guide_model_save_reader_en` for more details.

    Args:
        dirname(str): The given directory path.
        executor(Executor): The executor to run for loading inference model.
                            See :ref:`api_guide_executor_en` for more details about it.
        model_filename(str, optional): The name of file to load the inference program.
                                  If it is None, the default filename
                                  ``__model__`` will be used.
                                  Default: ``None``.
        params_filename(str, optional): The name of file to load all parameters.
                                   It is only used for the case that all
                                   parameters were saved in a single binary
                                   file. If parameters were saved in separate
                                   files, set it as ``None``.
                                   Default: ``None``.

        pserver_endpoints(list, optional): It is only needed by the distributed inference.
                                    If using a distributed look up table during the training,
                                    this table is also needed by the inference process. Its value is
                                    a list of pserver endpoints.

    Returns:
        list: The return of this API is a list with three elements:
        (program, feed_target_names, fetch_targets). The `program` is a
        ``Program`` (refer to :ref:`api_guide_Program_en`), which is used for inference.
        The `feed_target_names` is a list of ``str``, which contains names of variables
        that need to feed data in the inference program. The `fetch_targets` is a list of
        ``Variable`` (refer to :ref:`api_guide_Program_en`). It contains variables from which
        we can get inference results.

    Raises:
        ValueError: If `dirname` is not a existing directory.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            # Build the model
            main_prog = fluid.Program()
            startup_prog = fluid.Program()
            with fluid.program_guard(main_prog, startup_prog):
                data = fluid.layers.data(name="img", shape=[64, 784], append_batch_size=False)
                w = fluid.layers.create_parameter(shape=[784, 200], dtype='float32')
                b = fluid.layers.create_parameter(shape=[200], dtype='float32')
                hidden_w = fluid.layers.matmul(x=data, y=w)
                hidden_b = fluid.layers.elementwise_add(hidden_w, b)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_prog)

            # Save the inference model
            path = "./infer_model"
            fluid.io.save_inference_model(dirname=path, feeded_var_names=['img'],
                         target_vars=[hidden_b], executor=exe, main_program=main_prog)

            # Demo one. Not need to set the distributed look up table, because the
            # training doesn't use a distributed look up table.
            [inference_program, feed_target_names, fetch_targets] = (
                fluid.io.load_inference_model(dirname=path, executor=exe))
            tensor_img = np.array(np.random.random((1, 64, 784)), dtype=np.float32)
            results = exe.run(inference_program,
                          feed={feed_target_names[0]: tensor_img},
                          fetch_list=fetch_targets)

            # Demo two. If the training uses a distributed look up table, the pserver
            # endpoints list should be supported when loading the inference model.
            # The below is just an example.
            endpoints = ["127.0.0.1:2023","127.0.0.1:2024"]
            [dist_inference_program, dist_feed_target_names, dist_fetch_targets] = (
                fluid.io.load_inference_model(dirname=path,
                                              executor=exe,
                                              pserver_endpoints=endpoints))

            # In this example, the inference program was saved in the file
            # "./infer_model/__model__" and parameters were saved in
            # separate files under the directory "./infer_model".
            # By the inference program, feed_target_names and
            # fetch_targets, we can use an executor to run the inference
            # program for getting the inference result.
    """
    load_dirname = os.path.normpath(dirname)
    if not os.path.isdir(load_dirname):
        raise ValueError("There is no directory named '%s'", dirname)

    if model_filename is not None:
        model_filename = os.path.basename(model_filename)
    else:
        model_filename = "__model__"
    model_filename = os.path.join(load_dirname, model_filename)

    if params_filename is not None:
        params_filename = os.path.basename(params_filename)

    with open(model_filename, "rb") as f:
        program_desc_str = f.read()

    program = Program.parse_from_string(program_desc_str)
    if not core._is_program_version_supported(program._version()):
        raise ValueError("Unsupported program version: %d\n" %
                         program._version())
    # Binary data also need versioning.
    load_persistables(executor, load_dirname, program, params_filename)

    if pserver_endpoints:
        program = _endpoints_replacement(program, pserver_endpoints)

    feed_target_names = program.desc.get_feed_target_names()
    fetch_target_names = program.desc.get_fetch_target_names()
    fetch_targets = [
        program.global_block().var(name) for name in fetch_target_names
    ]

    return [program, feed_target_names, fetch_targets]


def _endpoints_replacement(program, endpoints):
    ENDPOINT_MAP = "epmap"
    for op in program.global_block().ops:
        if op.has_attr(ENDPOINT_MAP):
            op.set_attr(ENDPOINT_MAP, endpoints)
    program._sync_with_cpp()
    return program


def get_parameter_value(para, executor):
    """
    Get the LoDTensor value of the given parameter.

    Args:
        para(Parameter): The parameter to get value from.
        executor(Executor): The executor to run for retrieving the value.

    Returns:
        numpy.array: The given parameter's values.

    Raises:
        AssertionError: If the `para` is not an instance of Parameter.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            exe = fluid.Executor(fluid.CPUPlace())
            param = fluid.default_main_program().global_block().var('fc.w')
            p = fluid.io.get_parameter_value(param, exe)

    """
    assert is_parameter(para)

    get_program = Program()
    block = get_program.global_block()
    new_var = _clone_var_in_block_(block, para)
    return executor.run(get_program, feed={}, fetch_list=[new_var])[0]


def get_parameter_value_by_name(name, executor, program=None):
    """
    Get the LoDTensor value of a certain parameter by its name.

    Args:
        name(str): The parameter's name.
        executor(Executor): The executor to run for retrieving the value.
        program(Program | None): The program where to find the parameter.
                               If it's set to be None, the function will
                               try to find the parameter in the default
                               main program.

    Returns:
        numpy.array: The parameter's values.

    Raises:
        TypeError: If given `name` is not an instance of basestring.
        TypeError: If the parameter with the given name doesn't exist.
        AssertionError: If there is a variable named `name` in the
                        given program but it is not a Parameter.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            exe = fluid.Executor(fluid.CPUPlace())
            p = fluid.io.get_parameter_value('fc.w', exe)
    """
    if program is None:
        program = default_main_program()
    var = program.global_block().var(name)
    return get_parameter_value(var, executor)


def _save_persistable_nodes(executor, dirname, graph):
    """
    Save persistable nodes to the given directory by the executor.

    Args:
        executor(Executor): The executor to run for saving node values.
        dirname(str): The directory path.
        graph(IrGraph): All the required persistable nodes in the graph will be saved.
    """
    persistable_node_names = set()
    persistable_nodes = []
    all_persistable_nodes = graph.all_persistable_nodes()
    for node in all_persistable_nodes:
        name = cpt.to_text(node.name())
        if name not in persistable_node_names:
            persistable_node_names.add(name)
            persistable_nodes.append(node)
    program = Program()
    var_list = []
    for node in persistable_nodes:
        var_desc = node.var()
        if var_desc.type() == core.VarDesc.VarType.RAW or \
                var_desc.type() == core.VarDesc.VarType.READER:
            continue
        var = program.global_block().create_var(
            name=var_desc.name(),
            shape=var_desc.shape(),
            dtype=var_desc.dtype(),
            type=var_desc.type(),
            lod_level=var_desc.lod_level(),
            persistable=var_desc.persistable())
        var_list.append(var)
    save_vars(executor=executor, dirname=dirname, vars=var_list)


def _load_persistable_nodes(executor, dirname, graph):
    """
    Load persistable node values from the given directory by the executor.

    Args:
        executor(Executor): The executor to run for loading node values.
        dirname(str): The directory path.
        graph(IrGraph): All the required persistable nodes in the graph will be loaded.
    """
    persistable_node_names = set()
    persistable_nodes = []
    all_persistable_nodes = graph.all_persistable_nodes()
    for node in all_persistable_nodes:
        name = cpt.to_text(node.name())
        if name not in persistable_node_names:
            persistable_node_names.add(name)
            persistable_nodes.append(node)
    program = Program()
    var_list = []

    def _exist(var):
        return os.path.exists(os.path.join(dirname, var.name))

    for node in persistable_nodes:
        var_desc = node.var()
        if var_desc.type() == core.VarDesc.VarType.RAW or \
                var_desc.type() == core.VarDesc.VarType.READER:
            continue
        var = program.global_block().create_var(
            name=var_desc.name(),
            shape=var_desc.shape(),
            dtype=var_desc.dtype(),
            type=var_desc.type(),
            lod_level=var_desc.lod_level(),
            persistable=var_desc.persistable())
        if _exist(var):
            var_list.append(var)
        else:
            _logger.warn("Cannot find the var %s!!!" % (node.name()))
    load_vars(executor=executor, dirname=dirname, vars=var_list)


def save(program, model_path):
    """
    This function save parameters, optimizer information and network description to  model_path.

    The parameters contains all the trainable Variable, will save to a file with suffix ".pdparams".
    The optimizer information contains all the variable used by optimizer. For Adam optimizer, contains beta1, beta2, momentum etc. All the information will save to a file with suffix ".pdopt". (If the optimizer have no variable need to save (like SGD), the fill will not generated).
    The network description is the description of the program. It's only used for deployment. The description  will save to a file with a suffix ".pdmodel".
    
    Args:
        program(Program) : The program to saved.
        model_path(str): the file prefix to save the program. The format is "dirname/file_prefix". If file_prefix is empty str. A exception will be raised

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            prog = fluid.default_main_program()
            fluid.save( prog, "./temp")

    """

    base_name = os.path.basename(model_path)
    assert base_name != "", \
        "model_path MUST be format of dirname/filename [dirname\\filename in Window], Now filename is empty str"

    dir_name = os.path.dirname(model_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    def get_tensor(var):
        t = global_scope().find_var(var.name).get_tensor()
        return np.array(t)

    parameter_list = list(filter(is_parameter, program.list_vars()))
    param_dict = {p.name: get_tensor(p) for p in parameter_list}

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    with open(model_path + ".pdparams", 'wb') as f:
        pickle.dump(param_dict, f, protocol=2)

    optimizer_var_list = list(
        filter(is_belong_to_optimizer, program.list_vars()))

    opt_dict = {p.name: get_tensor(p) for p in optimizer_var_list}
    with open(model_path + ".pdopt", 'wb') as f:
        pickle.dump(opt_dict, f, protocol=2)

    main_program = program.clone()
    program.desc.flush()
    main_program.desc._set_version()
    paddle.fluid.core.save_op_compatible_info(program.desc)

    with open(model_path + ".pdmodel", "wb") as f:
        f.write(program.desc.serialize_to_string())


def load(program, model_path, executor=None, var_list=None):
    """
    This function get parameters and optimizer information from program, and then get corresponding value from file.
    An exception will throw if shape or dtype of the parameters is not match.

    This function can also load model file saved with [ save_params, save_persistables, save_vars ]. 
    var_list can not be None  when load single model file 
    ( filename is not None When save_params, save_persistables or save_vars is called ).

    Args: 
        program(Program): The program will be loaded
        model_path(str): The file prefix store the program
        executor(Executor, optional): The executor used for initialize the parameter 
                                      When startup program is not run.
        var_list(list, optional): The variable list to load single model file saved with 
                                  [ save_params, save_persistables, save_vars ]. 
                                  Default: None

    Returns:
        None
        
     Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            prog = fluid.default_main_program()
            fluid.save( prog, "./temp")

            fluid.load( prog, "./temp")

    """

    assert executor is None or isinstance(executor, Executor)

    model_prefix = model_path
    if model_prefix.endswith(".pdparams"):
        model_prefix = model_prefix[:-9]
    elif model_prefix.endswith(".pdopt"):
        model_prefix = model_prefix[:-6]
    elif model_prefix.endswith(".pdmodel"):
        model_prefix = model_prefix[:-8]

    parameter_file_name = model_prefix + ".pdparams"

    if not os.path.exists(parameter_file_name):
        # model file save by fluid.save not found, try to load model file saved with
        # [save_vars, save_params, save_persistables]
        _logger.warning(
            "{} not found, try to load model file saved with [ save_params, save_persistables, save_vars ]".
            format(parameter_file_name))
        if executor is None:
            raise ValueError(
                "executor is required when loading model file saved with [ save_params, save_persistables, save_vars ]"
            )
        if os.path.isdir(model_path):
            binary_file_set = set()
            for root, dirs, files in os.walk(model_path, topdown=False):
                for f in files:
                    binary_file_set.add(
                        os.path.join(root, f).replace("\\", "/"))
            program_var_list = list(program.list_vars())
            loaded_var_list = []
            for var in program_var_list:
                var_path = os.path.join(model_path, var.name).replace("\\", "/")
                if var_path in binary_file_set:
                    loaded_var_list.append(var)
                    binary_file_set.remove(var_path)
            if len(binary_file_set) > 0:
                unused_var_list = " ".join(list(binary_file_set))
                _logger.warning("variable file [ %s ] not used" %
                                (" ".join(list(binary_file_set))))
            try:
                load_vars(
                    executor=executor, dirname=model_path, vars=loaded_var_list)
            except RuntimeError as e:
                _logger.error(e)
                raise e
            except:
                raise RuntimeError(
                    "Failed to load model file , please make sure model file is saved with the "
                    "following APIs: save_params, save_persistables, save_vars")

            return
        elif os.path.isfile(model_path):
            if var_list == None:
                raise ValueError(
                    "var_list is required when loading model file saved with [ save_params, save_persistables, save_vars ]"
                )
            program_var_list = program.list_vars()
            program_var_name_set = set([var.name for var in program_var_list])

            # check all the variable inlcuded in program
            for var in var_list:
                if var.name not in program_var_name_set:
                    raise LookupError(
                        "loaded var [{}] not included in program variable list")

            dir_name, file_name = os.path.split(model_path)
            try:
                load_vars(
                    executor=executor,
                    dirname=dir_name,
                    vars=var_list,
                    filename=file_name)
            except RuntimeError as e:
                _logger.error(e)
                raise e
            except:
                raise RuntimeError( "Failed to load model file , please make sure model file is saved with the " \
                                    "the following APIs: [ save_params, save_persistables, save_vars ]. " \
                                    "When these API called, filename CANNOT be None")

            return

    def set_var(var, ndarray):
        t = global_scope().find_var(var.name).get_tensor()
        p = t._place()
        if p.is_cpu_place():
            place = paddle.fluid.CPUPlace()
        elif p.is_cuda_pinned_place():
            place = paddle.fluid.CUDAPinnedPlace()
        else:
            p = paddle.fluid.core.Place()
            p.set_place(t._place())
            place = paddle.fluid.CUDAPlace(p.gpu_device_id())

        t.set(ndarray, place)

    parameter_list = list(filter(is_parameter, program.list_vars()))

    if executor:
        paddle.fluid.core._create_loaded_parameter(parameter_list,
                                                   global_scope(),
                                                   executor._default_executor)
    with open(parameter_file_name, 'rb') as f:
        load_dict = pickle.load(f) if six.PY2 else pickle.load(
            f, encoding='latin1')
    for v in parameter_list:
        assert v.name in load_dict, \
            "Can not find [{}] in model file [{}]".format(
                v.name, parameter_file_name)
        set_var(v, load_dict[v.name])

    optimizer_var_list = list(
        filter(is_belong_to_optimizer, program.list_vars()))

    if len(optimizer_var_list) > 0:
        opt_file_name = model_prefix + ".pdopt"
        assert os.path.exists(opt_file_name), \
            "Optimizer file [{}] not exits".format(opt_file_name)

        if executor:
            paddle.fluid.core._create_loaded_parameter(
                optimizer_var_list, global_scope(), executor._default_executor)

        with open(opt_file_name, 'rb') as f:
            load_dict = pickle.load(f) if six.PY2 else pickle.load(
                f, encoding='latin1')
        for v in optimizer_var_list:
            assert v.name in load_dict, \
                "Can not find [{}] in model file [{}]".format(
                    v.name, opt_file_name)
            set_var(v, load_dict[v.name])


def load_program_state(model_path, var_list=None):
    """
    Load program state from local file
    
    Args:
        model_path(str): The file prefix store the program
        var_list(list, optional): The variable list to load saved with 
                                  [ save_params, save_persistables, save_vars ]. 
                                  Default: None.
                                  The var_list is only used to get name, 
                                  will not be modified.
    Returns:
        state_dict(dict): the dict store Parameter and optimizer information

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data( name="x", shape=[10, 10], dtype='float32')
            y = fluid.layers.fc( x, 10)
            z = fluid.layers.fc( y, 10)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run( fluid.default_startup_program() )
            prog = fluid.default_main_program()

            fluid.save( prog, "./temp")
            program_state = fluid.load_program_state( "./temp")
            
    """
    model_prefix = model_path
    if model_prefix.endswith(".pdparams"):
        model_prefix = model_prefix[:-9]
    elif model_prefix.endswith(".pdopt"):
        model_prefix = model_prefix[:-6]
    elif model_prefix.endswith(".pdmodel"):
        model_prefix = model_prefix[:-8]

    parameter_file_name = model_prefix + ".pdparams"
    if not os.path.exists(parameter_file_name):
        # model file saved with fluid.save is not found, try to load model file saved with
        # [save_vars, save_params, save_persistables]
        _logger.warning(
            "{} not found, try to load model file saved with [ save_params, save_persistables, save_vars ]".
            format(parameter_file_name))

        var_name_list = []
        if var_list is None and os.path.isfile(model_path):
            raise ValueError(
                "var_list can not be None when model_path is a file type")

        for root, dirs, files in os.walk(model_path, topdown=False):
            for f in files:
                file_path = os.path.join(root, f)
                var_temp_name = os.path.relpath(file_path, model_path)
                var_temp_name = var_temp_name.replace("\\", "/")
                var_name_list.append(var_temp_name)

        with _load_program_scope():
            load_prog = Program()
            load_block = load_prog.global_block()

            def clone_var_to_block(block, var):
                if not isinstance(var, Variable):
                    raise TypeError("value in var_list must be variable")
                return block.create_var(
                    name=var.name,
                    shape=var.shape,
                    dtype=var.dtype,
                    type=var.type,
                    lod_level=var.lod_level
                    if var.desc.type() == core.VarDesc.VarType.LOD_TENSOR else
                    None,
                    persistable=True)

            loaded_var_list = []

            if var_list is not None:
                for var in var_list:
                    loaded_var_list.append(clone_var_to_block(load_block, var))
            else:
                for var_name in var_name_list:
                    loaded_var_list.append(
                        load_block.create_var(
                            name=var_name, persistable=True))

            place = paddle.fluid.CPUPlace()
            exe = paddle.fluid.Executor(place)

            try:
                if os.path.isfile(model_path):
                    dir_name, file_name = os.path.split(model_path)
                else:
                    dir_name = model_path
                    file_name = None
                load_vars(
                    executor=exe,
                    dirname=dir_name,
                    vars=loaded_var_list,
                    filename=file_name)
            except:
                raise RuntimeError(
                    "Failed to load model file , please make sure model file is saved with the "
                    "following APIs: save_params, save_persistables, save_vars")
            res_dict = {}
            for var in loaded_var_list:
                res_dict[var.name] = np.asarray(paddle.fluid.global_scope(
                ).find_var(var.name).get_tensor())

            return res_dict

    assert os.path.exists(parameter_file_name), \
        "Parameter file [{}] not exits".format(parameter_file_name)

    with open(parameter_file_name, 'rb') as f:
        para_dict = pickle.load(f) if six.PY2 else pickle.load(
            f, encoding='latin1')

    opt_file_name = model_prefix + ".pdopt"
    if os.path.exists(opt_file_name):
        with open(opt_file_name, 'rb') as f:
            opti_dict = pickle.load(f) if six.PY2 else pickle.load(
                f, encoding='latin1')

        para_dict.update(opti_dict)

    return para_dict


def set_program_state(program, state_dict):
    """
    Set program parameter from state_dict

    An exception will throw if shape or dtype of the parameters is not match. 

    NOTICE: This function MUST called after run start_up_program

    Args:
        program(Program): The program to be set
        state_dict(dict): the dict store Parameter and optimizer information
    Returns: 
        None
    
    Examples:
        .. code-block:: python
            
            import paddle.fluid as fluid
            x = fluid.data( name="x", shape=[10, 10], dtype='float32')
            y = fluid.layers.fc( x, 10)
            z = fluid.layers.fc( y, 10)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run( fluid.default_startup_program() )
            prog = fluid.default_main_program()

            fluid.save( prog, "./temp")
            program_state = fluid.load_program_state( "./temp")

            fluid.set_program_state( prog, program_state)

    """
    parameter_list = list(filter(is_persistable, program.list_vars()))

    used_para_list = {}
    for para in parameter_list:
        var_temp = paddle.fluid.global_scope().find_var(para.name)
        assert var_temp != None, \
            "Variable [ {} ] Not found, Please make sure run startup program".format(para.name)
        if para.name in state_dict:
            # set value from state dict
            orig_para_np = np.array(var_temp.get_tensor())
            new_para_np = state_dict[para.name]
            assert orig_para_np.shape == new_para_np.shape, \
                "Shape not matching: the Program requires a parameter with a shape of ({}), " \
                "while the loaded parameter (namely [ {} ]) has a shape of  ({})." \
                    .format(orig_para_np.shape, para.name, new_para_np.shape)
            assert orig_para_np.dtype == new_para_np.dtype, \
                "Dtype not matching: the Program requires a parameter with a dtype of ({}), " \
                "while the loaded parameter (namely [ {} ]) has a dtype of  ({})." \
                    .format(orig_para_np.dtype, para.name, new_para_np.dtype)

            ten = var_temp.get_tensor()
            ten_place = ten._place()

            assert ten_place.is_gpu_place() or ten_place.is_cpu_place(), \
                "Place not support, only support CPUPlace and GPUPlace, now is {}".format(str(ten_place))
            py_place = paddle.fluid.CPUPlace()
            if ten_place.is_cuda_pinned_place():
                place = paddle.fluid.CUDAPinnedPlace()
            elif ten_place.is_gpu_place():
                p = paddle.fluid.core.Place()
                p.set_place(ten_place)
                py_place = paddle.fluid.CUDAPlace(p.gpu_device_id())

            ten.set(new_para_np, py_place)

            used_para_list[para.name] = 1

    unused_para_list = []
    for k, v in state_dict.items():
        if k not in used_para_list:
            unused_para_list.append(k)
    if len(unused_para_list) > 0:
        warnings.warn(
            "This list is not set, Because of Paramerter not found in program. There are: {}".
            format(" ".join(unused_para_list)))
