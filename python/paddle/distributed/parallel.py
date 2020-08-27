# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle import compat as cpt
from paddle.distributed.launch import get_cluster_and_pod, _print_arguments
from paddle.distributed.utils import _update_trainer_env

# deprecated module import
from paddle.fluid import core
from paddle.fluid.framework import _set_expected_place
from paddle.fluid.dygraph import parallel_helper
from paddle.fluid.dygraph.parallel import ParallelEnv

__all__ = ["init_parallel_env"]

ParallelStrategy = core.ParallelStrategy


# NOTE(chenweihang): The existence of this class leads to 
# the maintenance of two arguments. When the launch.py arguments 
# is updated, the arguments here also need to be updated, 
# but I have not thought of a better way here
class ParallelEnvArgs(object):
    def __init__(self):
        # Paddle cluster nodes ips, such as 192.168.0.16,192.168.0.17..
        self.cluster_node_ips = None

        # The current node ip.
        self.node_ip = None

        # whether to use paddlecloud platform to run your multi-process job.
        # If false, no need to set this argument.
        self.use_paddlecloud = None

        # The trainer's started port on a single node
        self.started_port = None

        # Print the config or not
        self.print_config = True

        # It's for gpu training and the training process will run 
        # on the selected_gpus, each process is bound to a single GPU. 
        # And if it's not set, this module will use all the gpu cards 
        # for training.
        self.selected_gpus = None


def _update_env_vars(rank, options):
    # 1. input check
    if not isinstance(rank, six.integer_types):
        raise TypeError("input `rank` type error, expected type is integer, "
                        "but received type is %s." % type(rank))
    if rank < 0:
        raise ValueError("input `rank` should be greater than 0, "
                         "but received %d." % rank)

    # 2. check and prepare environment variables
    # The necessary environment variables include:
    # - PADDLE_TRAINER_ID
    # - PADDLE_TRAINERS_NUM
    # - PADDLE_CURRENT_ENDPOINT
    # - PADDLE_TRAINER_ENDPOINTS

    # get args from kwargs
    args = ParallelEnvArgs()
    # set default `node_ip` and `cluster_node_ips`
    args.cluster_node_ips = options.get('cluster_node_ips', None)
    args.node_ip = options.get('node_ip', None)
    if args.cluster_node_ips is not None and args.node_ip is None:
        raise ValueError("please input current node ip, "
                         "cannot only give `cluster_node_ips`.")
    default_node_ip = os.environ.get("PADDLE_MASTER_IPADDR", None)
    default_node_ip = "127.0.0.1" if default_node_ip is None else default_node_ip
    if args.node_ip is None:
        args.node_ip = default_node_ip
    if args.cluster_node_ips is None:
        args.cluster_node_ips = default_node_ip

    # NOTE(chenweihang): Here should set `started_port` before
    # `get_cluster_and_pod` and keep each process's started_port
    # is same, see [ why need set default master info before run? ]
    args.started_port = options.get('started_port', None)
    if args.started_port is None:
        default_port = os.environ.get("PADDLE_MASTER_PORT", None)
        if default_port is None:
            raise ValueError(
                "Data parallel training start failed. If you start data parallel "
                "training by `paddle.distributed.launch` module, Please ensure "
                "that one of the following rules is met:\n"
                "  1. Do not set `paddle.distributed.init_parallel_env` argument "
                "`rank` or set it to be -1;\n"
                "  2. Set `paddle.distributed.init_parallel_env` start port for "
                "parallel training by `started_port=**`, e.g. started_port=6170."
            )
        args.started_port = int(default_port)

    args.use_paddlecloud = options.get('use_paddlecloud', False)
    args.print_config = options.get('print_config', True)

    # set default `selected_gpus`
    # TODO(chenweihang): if users given number of `selected_gpus`
    # is not equal to the spawn's nprocs, it will cause error, 
    # and because we remove the `proc num` argument of 
    # `init_parallel_env`, when above error occured, we do not 
    # have a good way to check, so users are not recommended to 
    # use this parameter, it is best to delete
    args.selected_gpus = options.get('selected_gpus', None)
    if args.selected_gpus is None:
        args.selected_gpus = os.environ.get("PADDLE_CUDA_VISIBLE_DEVICES", None)
        if args.selected_gpus is None:
            raise ValueError(
                "Data parallel training start failed. If you start data parallel "
                "training by `paddle.distributed.launch` module, Please ensure "
                "that one of the following rules is met:\n"
                "  1. Do not set `paddle.distributed.init_parallel_env` argument "
                "`rank` or set it to be -1;\n"
                "  2. Set `paddle.distributed.init_parallel_env` selected gpus of "
                "parallel training by `selected_gpus=**`, e.g. selected_gpus='0,1,2,3'."
            )

    # reuse code of launch.py
    cluster, pod = get_cluster_and_pod(args)

    # remove useless env vars
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)

    # update env vars
    trainer = pod.get_trainer(rank)
    if trainer is None:
        raise RuntimeError(
            "The expected trainer is not exists, its trainer rank is %d." %
            rank)
    # no copy, each process will hold env vars itself
    _update_trainer_env(os.environ, cluster, trainer)

    # print config
    if args.print_config and rank == 0:
        _print_arguments(args)


def _check_env_vars():
    def _check_var_exists(var_name):
        var = os.environ.get(var_name, None)
        if var is None:
            raise ValueError("paddle.distributed initialize error,"
                             "environment variable %s is needed, but not set." %
                             var_name)

    _check_var_exists("FLAGS_selected_gpus")
    _check_var_exists("PADDLE_TRAINER_ID")
    _check_var_exists("PADDLE_CURRENT_ENDPOINT")
    _check_var_exists("PADDLE_TRAINERS_NUM")
    _check_var_exists("PADDLE_TRAINER_ENDPOINTS")


def init_parallel_env(rank=-1, backend='nccl', **options):
    """
    Initialize parallel training environments in dynamic mode.

    Args:
        rank(int, optional): Rank of current process. Default value is -1. 
            When it is the default value -1, you should use ``paddle.distributed.launch`` 
            module to start training, the environment variables for parallel training 
            are configured by ``paddle.distributed.launch`` module.
        backend(str, optional): The backend to communication between multiple devices.
            Now only support ``nccl`` . Default value is ``nccl`` .
        **options(dict, optional): Other initial parallel execution environment configuration options.
            The following options are currently supported:

            - cluster_node_ips: Paddle cluster nodes ips, such as "192.168.0.16,192.168.0.17". Default: "127.0.0.1".

            - node_ip: The current node ip, such as "192.168.0.16". Default: "127.0.0.1".

            - started_port: The trainer's started port on a single node, such as 6170. Default: None.

            - selected_gpus: The training process will run on the selected_gpus, such as "0,1,2,3". Default: None.

            - print_config: Print current parallel training config. Default: True.

            - use_paddlecloud: Whether to use paddlecloud platform to run your multi-process job. Default: False.

    Returns:
        None
        
    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn
            import paddle.optimizer as opt
            import paddle.distributed as dist

            class LinearNet(nn.Layer):
                def __init__(self):
                    super(LinearNet, self).__init__()
                    self._linear1 = nn.Linear(10, 10)
                    self._linear2 = nn.Linear(10, 1)
                    
                def forward(self, x):
                    return self._linear2(self._linear1(x))

            def train(rank):
                # 1. enable dynamic mode
                paddle.disable_static()
                
                # 2. initialize parallel environment
                dist.init_parallel_env(rank)

                # 3. create data parallel layer & optimizer
                layer = LinearNet()
                dp_layer = paddle.DataParallel(layer)

                loss_fn = nn.MSELoss()
                sgd = opt.SGD(
                    learning_rate=0.001, parameter_list=dp_layer.parameters())

                # 4. run layer
                inputs = paddle.randn([10, 10], 'float32')
                outputs = dp_layer(inputs)
                labels = paddle.randn([10, 1], 'float32')
                loss = loss_fn(outputs, labels)
                
                loss = dp_layer.scale_loss(loss)
                loss.backward()
                dp_layer.apply_collective_grads()

                sgd.minimize(loss)
                dp_layer.clear_gradients()

            if __name__ == '__main__':
                dist.spawn(train, args=(), nprocs=2)
    """

    # 1. input check
    if not isinstance(backend, six.string_types):
        raise TypeError("input `backend` type error, expected type is str, "
                        "but received type is %s." % type(backend))
    if cpt.to_text(backend) != 'nccl':
        raise ValueError(
            "backend `%s` is not supported, now only supports `nccl` backend." %
            backend)

    # 2. update or check env
    # NOTE(chenweihang): if rank is default value, users should config 
    # parallel environment by module `paddle.distributed.launch`,
    # so here we only check the environment variables
    if rank != -1:
        _update_env_vars(rank, options)
    else:
        _check_env_vars()

    # 3. init ParallelStrategy
    strategy = ParallelStrategy()
    if cpt.to_text(backend) == 'nccl':
        strategy.nranks = ParallelEnv().world_size
        strategy.local_rank = ParallelEnv().rank
        strategy.trainer_endpoints = ParallelEnv().trainer_endpoints
        strategy.current_endpoint = ParallelEnv().current_endpoint
        if strategy.nranks < 2:
            return
        # NOTE(chenweihang): [ why config global place here? ]
        # the dygraph mode will be set to default mode, 
        # users will not call `dygraph.guard` or `enable_dygraph`
        # directly, if they want to switch default place,
        # they need to call a function to change default place,
        # here just set correctly place to users
        place = core.CUDAPlace(ParallelEnv().dev_id)
        _set_expected_place(place)

        # init nccl context
        parallel_helper._set_parallel_ctx(
            core.NCCLParallelContext(strategy, place))
        parallel_helper._init_parallel_ctx()
