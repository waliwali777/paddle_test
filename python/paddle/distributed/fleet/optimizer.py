#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import warnings
import paddle
import os
import numpy as np
from paddle.fluid.framework import dygraph_only, _global_flags
from .base.distributed_strategy import DistributedStrategy
from .meta_optimizers import HybridParallelOptimizer, HeterParallelOptimizer
from paddle.fluid import core
from paddle.distributed import fleet


def distributed_optimizer(optimizer, strategy=None):
    """
        Optimizer for distributed training.
        For the distributed training, this method would rebuild a new instance of DistributedOptimizer.
        Which has basic Optimizer function and special features for distributed training.
        Args:
            optimizer(Optimizer): The executor to run for init server.
            strategy(DistributedStrategy): Extra properties for distributed optimizer.
                It is recommended to use DistributedStrategy in fleet.init(). The strategy
                here is for compatibility. If the strategy in fleet.distributed_optimizer()
                is not None, then it will overwrite the DistributedStrategy in fleet.init(),
                which will take effect in distributed training.
        Returns:
            Fleet: instance of fleet.
        Examples:
            .. code-block:: python
                import paddle
                import paddle.distributed.fleet as fleet
                fleet.init(is_collective=True)
                strategy = fleet.DistributedStrategy()
                optimizer = paddle.optimizer.SGD(learning_rate=0.001)
                optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        """
    fleet.user_defined_optimizer = optimizer

    fleet.step = optimizer.step
    fleet.clear_grad = optimizer.clear_grad
    fleet.set_lr = optimizer.set_lr
    fleet.get_lr = optimizer.get_lr
    fleet.state_dict = optimizer.state_dict
    fleet.set_state_dict = optimizer.set_state_dict

    fleet_env = fleet.fleet
    if strategy is not None:
        if fleet_env._is_collective:
            warnings.warn(
                "It is recommended to use DistributedStrategy "
                "in fleet_env.init(). The strategy here is only for compatibility. "
                "If the strategy in fleet_env.distributed_optimizer() is "
                "not None, then it will overwrite the DistributedStrategy in fleet_env.init(), "
                "which will take effect in distributed training.")
        fleet_env._user_defined_strategy = copy.deepcopy(strategy)

    fleet_env._context = {}

    if fleet_env.worker_num() > 1:
        if fleet_env._user_defined_strategy.heter_ccl_mode == False:
            return HybridParallelOptimizer(optimizer, fleet_env._hcg,
                                           fleet_env._user_defined_strategy)
        else:
            return HeterParallelOptimizer(optimizer,
                                          fleet_env._user_defined_strategy)
    else:
        return optimizer
