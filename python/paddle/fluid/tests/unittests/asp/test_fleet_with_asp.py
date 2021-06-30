# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
import unittest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import os
from paddle.fluid.contrib import sparsity
from paddle.fluid.contrib.sparsity.asp import ASPHelper
import numpy as np

paddle.enable_static()


class TestFleetWithASP(unittest.TestCase):
    def setUp(self):
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    #     os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001"

    def net(self, main_prog, startup_prog):
        with fluid.program_guard(main_prog, startup_prog):
            img = fluid.data(
                name='img', shape=[None, 3, 32, 32], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            hidden = fluid.layers.conv2d(
                input=img, num_filters=4, filter_size=3, padding=2, act="relu")
            hidden = fluid.layers.fc(input=hidden, size=32, act='relu')
            prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')

            cost = paddle.fluid.layers.cross_entropy(
                input=prediction, label=label)
            avg_cost = paddle.fluid.layers.mean(x=cost)

            strategy = paddle.distributed.fleet.DistributedStrategy()
            strategy.asp = True
        return avg_cost, strategy, img, label

    def test_with_asp(self):
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        train_prog, startup_prog = fluid.Program(), fluid.Program()
        avg_cost, strategy, input_x, input_y = self.net(train_prog,
                                                        startup_prog)

        with fluid.program_guard(train_prog, startup_prog):
            optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.01)
            optimizer = fleet.distributed_optimizer(
                optimizer, strategy=strategy)
            optimizer.minimize(avg_cost)

        place = paddle.CPUPlace()
        if core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(feed_list=[input_x, input_y], place=place)
        exe.run(startup_prog)

        sparsity.prune_model(place, train_prog)

        data = (np.random.randn(64, 3, 32, 32), np.random.randint(
            10, size=(64, 1)))
        exe.run(train_prog, feed=feeder.feed([data]))

        for param in train_prog.global_block().all_parameters():
            if ASPHelper._is_supported_layer(train_prog, param.name):
                mat = np.array(fluid.global_scope().find_var(param.name)
                               .get_tensor())
                self.assertTrue(sparsity.check_sparsity(mat.T, n=2, m=4))


if __name__ == "__main__":
    unittest.main()
