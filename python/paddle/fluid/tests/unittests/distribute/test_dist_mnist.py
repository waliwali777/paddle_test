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

import numpy as np
import argparse
import time
import math

import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
from paddle.fluid import core
import unittest
from multiprocessing import Process
import os
import signal

from dist_test import FluidDistTest

SEED = 1
DTYPE = "float32"
paddle.dataset.mnist.fetch()


# random seed must set before configuring the network.
# fluid.default_startup_program().random_seed = SEED
def cnn_model(data):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=data,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")

    # TODO(dzhwinter) : refine the initializer and random seed settting
    SIZE = 10
    input_shape = conv_pool_2.shape
    param_shape = [reduce(lambda a, b: a * b, input_shape[1:], 1)] + [SIZE]
    scale = (2.0 / (param_shape[0]**2 * SIZE))**0.5

    predict = fluid.layers.fc(
        input=conv_pool_2,
        size=SIZE,
        act="softmax",
        param_attr=fluid.param_attr.ParamAttr(
            initializer=fluid.initializer.NormalInitializer(
                loc=0.0, scale=scale)))
    return predict


def get_model(batch_size):
    # Input data
    images = fluid.layers.data(name='pixel', shape=[1, 28, 28], dtype=DTYPE)
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # Train program
    predict = cnn_model(images)
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    # Evaluator
    batch_size_tensor = fluid.layers.create_tensor(dtype='int64')
    batch_acc = fluid.layers.accuracy(
        input=predict, label=label, total=batch_size_tensor)

    inference_program = fluid.default_main_program().clone()
    # Optimization
    opt = fluid.optimizer.AdamOptimizer(
        learning_rate=0.001, beta1=0.9, beta2=0.999)

    # Reader
    train_reader = paddle.batch(
        paddle.dataset.mnist.train(), batch_size=batch_size)
    test_reader = paddle.batch(
        paddle.dataset.mnist.test(), batch_size=batch_size)
    opt.minimize(avg_cost)
    return inference_program, avg_cost, train_reader, test_reader, batch_acc


def get_transpiler(trainer_id, main_program, pserver_endpoints, trainers):
    t = fluid.DistributeTranspiler()
    t.transpile(
        trainer_id=trainer_id,
        program=main_program,
        pservers=pserver_endpoints,
        trainers=trainers)
    return t


class TestDistMnist(FluidDistTest):
    def setUp(self):
        self.place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        self.entry = "python test_dist_mnist.py"

    def test_job(self):
        self.launch_job(2, 2)

    def start_pserver(self):
        ps_endpoints = os.getenv("PADDLE_PSERVER_ENDPOINTS")
        trainers = int(os.getenv("PADDLE_TRAINERS"))
        current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")

        get_model(batch_size=20)
        t = get_transpiler(0,
                           fluid.default_main_program(), ps_endpoints, trainers)
        pserver_prog = t.get_pserver_program(current_endpoint)
        startup_prog = t.get_startup_program(current_endpoint, pserver_prog)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_prog)

        exe.run(pserver_prog)

    def start_trainer(self):
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
        ps_endpoints = os.getenv("PADDLE_PSERVER_ENDPOINTS")
        trainers = int(os.getenv("PADDLE_TRAINERS"))

        test_program, avg_cost, train_reader, test_reader, batch_acc = get_model(
            batch_size=20)
        t = get_transpiler(trainer_id,
                           fluid.default_main_program(), ps_endpoints, trainers)

        trainer_prog = t.get_trainer_program()

        exe = fluid.Executor(self.place)
        exe.run(fluid.default_startup_program())

        feed_var_list = [
            var for var in trainer_prog.global_block().vars.itervalues()
            if var.is_data
        ]

        feeder = fluid.DataFeeder(feed_var_list, self.place)
        for pass_id in xrange(10):
            for batch_id, data in enumerate(train_reader()):
                exe.run(trainer_prog, feed=feeder.feed(data))

                if (batch_id + 1) % 10 == 0:
                    acc_set = []
                    avg_loss_set = []
                    for test_data in test_reader():
                        acc_np, avg_loss_np = exe.run(
                            program=test_program,
                            feed=feeder.feed(test_data),
                            fetch_list=[batch_acc, avg_cost])
                        acc_set.append(float(acc_np))
                        avg_loss_set.append(float(avg_loss_np))
                    # get test acc and loss
                    acc_val = np.array(acc_set).mean()
                    avg_loss_val = np.array(avg_loss_set).mean()
                    if float(acc_val
                             ) > 0.8:  # Smaller value to increase CI speed
                        return
                    else:
                        print(
                            'TRAINER_ID: {0} PassID {1}, BatchID {2}, Test Loss {3:2.2}, Acc {4:2.2}'.
                            format(trainer_id, pass_id, batch_id + 1,
                                   float(avg_loss_val), float(acc_val)))
                        if math.isnan(float(avg_loss_val)):
                            assert ("got Nan loss, training failed.")


if __name__ == "__main__":
    unittest.main()
