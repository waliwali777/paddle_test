# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
import numpy as np
from test_dist_base import TestDistRunnerBase, runtime_main
from dist_mnist import cnn_model


class TestDistMnistGradientMergeRawOptimizer(TestDistRunnerBase):
    def get_model(self, batch_size=2, single_device=False):
        paddle.enable_static()
        paddle.seed(1)
        np.random.seed(1)

        assert fluid.core.globals()['FLAGS_apply_pass_to_program']
        strategy = fleet.DistributedStrategy()
        strategy.gradient_merge = True
        strategy.gradient_merge_configs = {
            "k_steps": 2,
            "avg": False,
        }
        strategy.without_graph_optimization = True

        fleet.init(is_collective=True, strategy=strategy)
        image = paddle.static.data(
            name='image', shape=[None, 1, 28, 28], dtype="float32")
        label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
        predict = cnn_model(image)
        acc = paddle.metric.accuracy(predict, label)
        loss_fn = nn.CrossEntropyLoss(use_softmax=False)
        cost = loss_fn(predict, label)
        test_program = paddle.static.default_main_program().clone(for_test=True)
        optimizer = paddle.optimizer.Adam(learning_rate=1e-3)
        if single_device:
            optimizer = fluid.optimizer.GradientMergeOptimizer(
                optimizer,
                k_steps=strategy.gradient_merge_configs["k_steps"],
                avg=strategy.gradient_merge_configs["avg"])
        else:
            optimizer = fleet.distributed_optimizer(optimizer)
        optimizer.minimize(cost)
        train_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size)
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size)
        return test_program, cost, train_reader, test_reader, acc, predict


if __name__ == "__main__":
    runtime_main(TestDistMnistGradientMergeRawOptimizer)
