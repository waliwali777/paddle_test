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

import paddle.fluid as fluid
import paddle.fluid.framework as framework
from paddle.fluid.distribute_transpiler import delete_ops


def train_network():
    y_predict = fluid.layers.fc(input=x, size=1, act=None)

    y = fluid.layers.data(name='y', shape=[1], dtype='float32')
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    optimize_ops, params_grads = sgd_optimizer.minimize(avg_cost)
    return optimize_ops, params_grads


def save_program_desc(network_func):
    startup_program = framework.Program()
    train_program = framework.Program()

    with framework.program_guard(train_program, startup_program):
        optimize_ops, params_grads = network_func()
        delete_ops(train_program.global_block(), optimize_ops)

    with open("startup_program", "w") as f:
        f.write(startup_program.desc.serialize_to_string())
    with open("train_program", "w") as f:
        f.write(train_program.desc.serialize_to_string())
