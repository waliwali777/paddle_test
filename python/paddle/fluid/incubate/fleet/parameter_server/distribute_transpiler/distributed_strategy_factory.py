#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

__all__ = [
    "TrainerRuntimeConfig", "DistributedStrategy", "DistributedStrategyFactory"
]
import os
import paddle.fluid as fluid
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig, ServerRuntimeConfig


class TrainerRuntimeConfig(object):
    def __init__(self):
        self._communicator_flags = dict()
        self._communicator_flags["max_merge_var_num"] = int(
            os.getenv("FLAGS_communicator_max_merge_var_num", "20"))
        self._communicator_flags["send_queue_size"] = int(
            os.getenv("FLAGS_communicator_send_queue_size", "20"))
        self._communicator_flags["independent_recv_thread"] = bool(
            int(os.getenv("FLAGS_communicator_independent_recv_thread", "1")))
        self._communicator_flags["min_send_grad_num_before_recv"] = int(
            os.getenv("FLAGS_communicator_min_send_grad_num_before_recv", "20"))
        self._communicator_flags["thread_pool_size"] = int(
            os.getenv("FLAGS_communicator_thread_pool_size", "5"))
        self._communicator_flags["send_wait_times"] = int(
            os.getenv("FLAGS_communicator_send_wait_times", "5"))
        self._communicator_flags["fake_rpc"] = int(
            os.getenv("FLAGS_communicator_fake_rpc", "0"))
        self._communicator_flags["merge_sparse_grad"] = int(
            os.getenv("FLAGS_communicator_merge_sparse_grad", "1"))
        self._communicator_flags["is_sgd_optimizer"] = int(
            os.getenv("communicator_is_sgd_optimizer", "1"))

        self._rpc_deadline = int(os.getenv("FLAGS_rpc_deadline", "180000"))
        self._rpc_retry_times = int(os.getenv("FLAGS_rpc_retry_times", "3"))

    def __str__(self):
        print_str = "communicator_max_merge_var_num: {}\n" % self._communicator_flags[
            "max_merge_var_num"]
        print_str += "communicator_send_queue_size: {}\n" % self._communicator_flags[
            "send_queue_size"]
        print_str += "communicator_independent_recv_thread: {}\n" % self._communicator_flags[
            "independent_recv_thread"]
        print_str += "communicator_min_send_grad_num_before_recv: {}\n" % self._communicator_flags[
            "min_send_grad_num_before_recv"]
        print_str += "communicator_thread_pool_size: {}\n" % self._communicator_flags[
            "thread_pool_size"]
        print_str += "communicator_send_wait_times: {}\n" % self._communicator_flags[
            "send_wait_times"]
        print_str += "communicator_fake_rpc: {}\n" % self._communicator_flags[
            "fake_rpc"]
        print_str += "communicator_merge_sparse_grad: {}\n" % self._communicator_flags[
            "merge_sparse_grad"]
        print_str += "rpc_deadline: {}\n" % self._rpc_deadline
        print_str += "rpc_retry_times: {}" % self._rpc_retry_times
        return print_str

    def __repr__(self):
        return self.__str__()


class DistributedStrategy(object):
    def __init__(self):
        self._program_config = DistributeTranspilerConfig()
        self._trainer_runtime_config = TrainerRuntimeConfig()
        self._server_runtime_config = ServerRuntimeConfig()
        self._execute_strategy = fluid.ExecutionStrategy()
        self._build_strategy = fluid.BuildStrategy()
        num_threads = int(os.getenv("CPU_NUM", "1"))
        self._execute_strategy.num_threads = num_threads
        if num_threads > 1:
            self._build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce

    def get_program_config(self):
        return self._program_config

    def set_program_config(self, config):
        if isinstance(config, DistributeTranspilerConfig):
            self._program_config = config
        elif isinstance(config, dict):
            for key in config:
                if hasattr(self._program_config, key):
                    setattr(self._program_config, key, config[key])
                else:
                    raise ValueError(
                        "DistributeTranspilerConfig doesn't have key: '%s'",
                        key)
        else:
            raise TypeError(
                "program_config only accept input type: dict or DistributeTranspilerConfig"
            )

    def get_trainer_runtime_config(self):
        return self._trainer_runtime_config

    def set_trainer_runtime_config(self, config):
        if isinstance(config, TrainerRuntimeConfig):
            self._trainer_runtime_config = config
        elif isinstance(config, dict):
            for key in config:
                if hasattr(self._trainer_runtime_config, key):
                    setattr(self._trainer_runtime_config, key, config[key])
                elif key in self._trainer_runtime_config._communicator_flags:
                    self._trainer_runtime_config._communicator_flags[key] = int(config[key])
                else:
                    raise ValueError(
                        "TrainerRuntimeConfig doesn't have key: '%s'", key)
        else:
            raise TypeError(
                "trainer_runtime_config only accept input type: dict or TrainerRuntimeConfig"
            )

    def get_server_runtime_config(self):
        return self._server_runtime_config

    def set_server_runtime_config(self, config):
        if isinstance(config, ServerRuntimeConfig):
            self._server_runtime_config = config
        elif isinstance(config, dict):
            for key in config:
                if hasattr(self._server_runtime_config, key):
                    setattr(self._server_runtime_config, key, config[key])
                else:
                    raise ValueError(
                        "ServerRuntimeConfig doesn't have key: '%s'", key)
        else:
            raise TypeError(
                "server_runtime_config only accept input type: dict or ServerRuntimeConfig"
            )

    def get_execute_strategy(self):
        return self._execute_strategy

    def set_execute_strategy(self, config):
        if isinstance(config, fluid.ExecutionStrategy):
            self._execute_strategy = config
        elif isinstance(config, dict):
            for key in config:
                if hasattr(self._execute_strategy, key):
                    setattr(self._execute_strategy, key, config[key])
                else:
                    raise ValueError("ExecutionStrategy doesn't have key: '%s'",
                                     key)
        else:
            raise TypeError(
                "execute_strategy only accept input type: dict or ExecutionStrategy"
            )

    def get_build_trategy(self):
        return self._build_strategy

    def set_build_strategy(self, config):
        if isinstance(config, fluid.BuildStrategy):
            self._build_strategy = config
        elif isinstance(config, dict):
            for key in config:
                if hasattr(self._build_strategy, key):
                    setattr(self._build_strategy, key, config[key])
                else:
                    raise ValueError("BuildStrategy doesn't have key: '%s'",
                                     key)
        else:
            raise TypeError(
                "build_strategy only accept input type: dict or BuildStrategy")


class SyncStrategy(DistributedStrategy):
    def __init__(self):
        super(SyncStrategy, self).__init__()
        self._program_config.sync_mode = True
        self._program_config.runtime_split_send_recv = False
        self._build_strategy.async_mode = False


class AsyncStrategy(DistributedStrategy):
    def __init__(self):
        super(AsyncStrategy, self).__init__()
        self._program_config.sync_mode = False
        self._program_config.runtime_split_send_recv = True
        self._build_strategy.async_mode = True


class HalfAsyncStrategy(DistributedStrategy):
    def __init__(self):
        super(HalfAsyncStrategy, self).__init__()
        self._program_config.sync_mode = False
        self._program_config.runtime_split_send_recv = False
        self._build_strategy.async_mode = False


class GeoStrategy(DistributedStrategy):
    def __init__(self, update_frequency=100):
        super(GeoStrategy, self).__init__()
        self._program_config.sync_mode = False
        self._program_config.runtime_split_send_recv = True
        self._program_config.geo_sgd_mode = True
        self._program_config.geo_sgd_need_push_nums = update_frequency
        self._build_strategy.async_mode = True


class DistributedStrategyFactory(object):
    def __init_(self):
        self._distributed_strategy = None

    def create_sync_strategy(self):
        self._distributed_strategy = SyncStrategy()
        return self._distributed_strategy

    def create_half_async_strategy(self):
        self._distributed_strategy = HalfAsyncStrategy()
        return self._distributed_strategy

    def create_async_strategy(self):
        self._distributed_strategy = AsyncStrategy()
        return self._distributed_strategy

    def create_geo_strategy(self, update_frequency=100):
        self._distributed_strategy = GeoStrategy(update_frequency)
        return self._distributed_strategy
