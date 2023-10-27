# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os

from semi_auto_parallel_simple_net import (
    CLASS_NUM,
    IMAGE_SIZE,
    TestSimpleNetForSemiAutoParallel,
)

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import nn


class DPAndMPDemoNet(nn.Layer):
    def __init__(self, np_w0, np_w1, mesh):
        super().__init__()
        self.mesh = mesh
        self.w0 = dist.shard_tensor(
            self.create_parameter(
                shape=[IMAGE_SIZE, IMAGE_SIZE],
                attr=paddle.framework.ParamAttr(
                    name="dmp_demo_weight_1",
                    initializer=paddle.nn.initializer.Assign(np_w0),
                ),
            ),
            dist_attr=dist.DistAttr(mesh=mesh, sharding_specs=[None, 'y']),
        )
        self.w1 = dist.shard_tensor(
            self.create_parameter(
                shape=[IMAGE_SIZE, CLASS_NUM],
                attr=paddle.framework.ParamAttr(
                    name="dmp_nemo_weight_2",
                    initializer=paddle.nn.initializer.Assign(np_w1),
                ),
            ),
            dist_attr=dist.DistAttr(mesh=mesh, sharding_specs=['y', None]),
        )

    def forward(self, x):
        out = F.linear(
            dist.shard_tensor(
                x,
                dist_attr=dist.DistAttr(
                    mesh=self.mesh, sharding_specs=['x', None]
                ),
            ),
            self.w0,
        )
        out = F.relu(out)
        out = F.linear(out, self.w1)

        return out


class DPAndMPAndPPDemoNet(nn.Layer):
    def __init__(self, np_w0, np_w1, mesh0, mesh1):
        super().__init__()
        self.mesh0 = mesh0
        self.mesh1 = mesh1
        self.w0 = dist.shard_tensor(
            self.create_parameter(
                shape=[IMAGE_SIZE, IMAGE_SIZE],
                attr=paddle.framework.ParamAttr(
                    name="dmpp_demo_weight_1",
                    initializer=paddle.nn.initializer.Assign(np_w0),
                ),
            ),
            dist_attr=dist.DistAttr(mesh=mesh0, sharding_specs=[None, 'y']),
        )
        self.w1 = dist.shard_tensor(
            self.create_parameter(
                shape=[IMAGE_SIZE, CLASS_NUM],
                attr=paddle.framework.ParamAttr(
                    name="dmpp_nemo_weight_2",
                    initializer=paddle.nn.initializer.Assign(np_w1),
                ),
            ),
            dist_attr=dist.DistAttr(mesh=mesh1, sharding_specs=['y', None]),
        )

    def forward(self, x):
        out = F.linear(
            dist.shard_tensor(
                x,
                dist_attr=dist.DistAttr(
                    mesh=self.mesh0, sharding_specs=['x', None]
                ),
            ),
            self.w0,
        )
        out = F.relu(out)
        out = dist.reshard(
            out,
            dist_attr=dist.DistAttr(mesh=self.mesh1, sharding_specs=['x', 'y']),
        )
        out = F.linear(out, self.w1)

        return out


class TestSimpleNetHybridStrategyForSemiAutoParallel(
    TestSimpleNetForSemiAutoParallel
):
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])
        self._mesh0 = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])
        self._mesh1 = dist.ProcessMesh([[4, 5], [6, 7]], dim_names=["x", "y"])

        paddle.set_device(self._backend)

        self.init_input_data()
        self.init_single_card_net_result()

    def test_dp_mp_demo_net(self):
        (
            self.dp_mp_loss,
            self.dp_mp_w0,
            self.dp_mp_w1,
        ) = self.run_dynamic(DPAndMPDemoNet(self.w0, self.w1, self._mesh))
        self.check_tensor_eq(self.dp_mp_loss, self.base_loss)
        self.check_tensor_eq(self.dp_mp_w0, self.base_w0)
        self.check_tensor_eq(self.dp_mp_w1, self.base_w1)
        self.check_tensor_eq(self.dp_mp_w0.grad, self.base_w0.grad)
        self.check_tensor_eq(self.dp_mp_w1.grad, self.base_w1.grad)

    def dp_mp_pp_demo_net(self):
        (
            self.dp_mp_pp_loss,
            self.dp_mp_pp_w0,
            self.dp_mp_pp_w1,
        ) = self.run_dynamic(
            DPAndMPAndPPDemoNet(self.w0, self.w1, self._mesh0, self._mesh1)
        )
        rank = dist.get_rank()

        if rank in [4, 5, 6, 7]:
            self.check_tensor_eq(self.dp_mp_pp_loss, self.base_loss)
            self.check_tensor_eq(self.dp_mp_pp_w1, self.base_w1)
            self.check_tensor_eq(self.dp_mp_pp_w1.grad, self.base_w1.grad)
        else:
            self.check_tensor_eq(self.dp_mp_pp_w0, self.base_w0)
            self.check_tensor_eq(self.dp_mp_pp_w0.grad, self.base_w0.grad)

    def run_test_case(self):
        self.test_dp_mp_demo_net()
        # TODO(GhostScreaming): Paddle-CI-Coverage doesn't support 8-cards
        # testcase now. Enable it later. It can be tested with
        # modify test_semi_auto_parallel_hybrid_strategy.py `setUp` function,
        # just set num_of_devices=8, nnode =1 and _changeable_envs = {"backend": ["gpu"]}
        # to test it.
        # self.dp_mp_pp_demo_net()


if __name__ == '__main__':
    TestSimpleNetHybridStrategyForSemiAutoParallel().run_test_case()
