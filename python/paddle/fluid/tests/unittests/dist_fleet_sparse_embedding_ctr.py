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
"""
Distribute CTR model for test fleet api
"""

from __future__ import print_function

import shutil
import tempfile
import time

import random
import numpy as np

import paddle
import paddle.fluid as fluid

from test_dist_fleet_base import runtime_main, FleetDistRunnerBase


def fake_ctr_reader():
    def reader():
        for _ in range(1000):
            deep = np.random.random_integers(0, 1e10, size=16).tolist()
            wide = np.random.random_integers(0, 1e10, size=8).tolist()
            label = np.random.random_integers(0, 1, size=1).tolist()
            yield [deep, wide, label]

    return reader


class TestDistCTR2x2(FleetDistRunnerBase):
    """
    For test CTR model, using Fleet api
    """

    def net(self, args, batch_size=4, lr=0.01):
        """
        network definition

        Args:
            batch_size(int): the size of mini-batch for training
            lr(float): learning rate of training
        Returns:
            avg_cost: LoDTensor of cost.
        """
        dnn_input_dim, lr_input_dim = 10, 10

        dnn_data = fluid.layers.data(
            name="dnn_data",
            shape=[-1, 1],
            dtype="int64",
            lod_level=1,
            append_batch_size=False)
        lr_data = fluid.layers.data(
            name="lr_data",
            shape=[-1, 1],
            dtype="int64",
            lod_level=1,
            append_batch_size=False)
        label = fluid.layers.data(
            name="click",
            shape=[-1, 1],
            dtype="int64",
            lod_level=0,
            append_batch_size=False)

        datas = [dnn_data, lr_data, label]

        if args.reader == "pyreader":
            self.reader = fluid.io.PyReader(
                feed_list=datas,
                capacity=64,
                iterable=False,
                use_double_buffer=False)

        # build dnn model
        dnn_layer_dims = [128, 64, 32]
        dnn_embedding = fluid.contrib.layers.sparse_embedding(
            input=dnn_data,
            size=[dnn_input_dim, dnn_layer_dims[0]],
            param_attr=fluid.ParamAttr(
                name="deep_embedding",
                initializer=fluid.initializer.Constant(value=0.01)))
        dnn_pool = fluid.layers.sequence_pool(
            input=dnn_embedding, pool_type="sum")
        dnn_out = dnn_pool
        for i, dim in enumerate(dnn_layer_dims[1:]):
            fc = fluid.layers.fc(
                input=dnn_out,
                size=dim,
                act="relu",
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(value=0.01)),
                name='dnn-fc-%d' % i)
            dnn_out = fc

        # build lr model
        lr_embbding = fluid.layers.embedding(
            is_distributed=False,
            input=lr_data,
            size=[lr_input_dim, 1],
            param_attr=fluid.ParamAttr(
                name="wide_embedding",
                initializer=fluid.initializer.Constant(value=0.01)),
            is_sparse=True)

        lr_pool = fluid.layers.sequence_pool(input=lr_embbding, pool_type="sum")

        merge_layer = fluid.layers.concat(input=[dnn_out, lr_pool], axis=1)
        predict = fluid.layers.fc(input=merge_layer, size=2, act='softmax')

        acc = fluid.layers.accuracy(input=predict, label=label)
        auc_var, _, _ = fluid.layers.auc(input=predict, label=label)
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = fluid.layers.mean(x=cost)

        self.feeds = datas
        self.train_file_path = ["fake1", "fake2"]
        self.avg_cost = avg_cost
        self.predict = predict

        return avg_cost

    def do_pyreader_training(self, fleet):
        """
        do training using dataset, using fetch handler to catch variable
        Args:
            fleet(Fleet api): the fleet object of Parameter Server, define distribute training role
        """

        exe = fluid.Executor(fluid.CPUPlace())

        fleet.init_worker()
        exe.run(fleet.startup_program)

        batch_size = 4

        train_reader = paddle.batch(fake_ctr_reader(), batch_size=batch_size)
        self.reader.decorate_sample_list_generator(train_reader)

        compiled_prog = fluid.compiler.CompiledProgram(
            fleet.main_program).with_data_parallel(
                loss_name=self.avg_cost.name,
                build_strategy=self.strategy.get_build_strategy(),
                exec_strategy=self.strategy.get_execute_strategy())

        for epoch_id in range(1):
            self.reader.start()
            try:
                pass_start = time.time()
                while True:
                    loss_val = exe.run(program=compiled_prog,
                                       fetch_list=[self.avg_cost.name])
                    loss_val = np.mean(loss_val)
                    print("TRAIN ---> pass: {} loss: {}\n".format(epoch_id,
                                                                  loss_val))
                pass_time = time.time() - pass_start
            except fluid.core.EOFException:
                self.reader.reset()

        model_dir = tempfile.mkdtemp()
        fleet.save_inference_model(
            exe, model_dir, [feed.name for feed in self.feeds], self.avg_cost)
        shutil.rmtree(model_dir)
        fleet.stop_worker()


if __name__ == "__main__":
    runtime_main(TestDistCTR2x2)
