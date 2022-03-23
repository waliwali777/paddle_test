# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import paddle
import numpy as np
from paddle.distributed.models.moe import utils
from paddle.fluid import core


def limit_by_capacity(expert_count, _capacity, n_worker):
    capacity = np.copy(_capacity)
    old_shape = expert_count.shape
    expert_count = np.reshape(expert_count, (n_worker, len(capacity)))
    output = np.zeros_like(expert_count)
    for wid in range(len(expert_count)):
        for eid in range(len(expert_count[wid])):
            last_cap = capacity[eid]
            if last_cap >= 0:
                capacity[eid] -= expert_count[wid][eid]
            if last_cap >= expert_count[wid][eid]:
                output[wid][eid] = expert_count[wid][eid]
            elif last_cap >= 0:
                output[wid][eid] = last_cap
    return output.reshape(old_shape)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestLimitByCapacityInt64API(unittest.TestCase):
    def init_test_case(self):
        self.capacity = np.array([100, 12000, 1200, 800, 4700, 10000, 57, 99])
        self.n_worker = 24
        self.expert_count = np.random.randint(
            0, 1000, size=(len(self.capacity) * self.n_worker))
        self.out = limit_by_capacity(self.expert_count, self.capacity,
                                     self.n_worker)
        self.place = paddle.CUDAPlace(0)

    def setUp(self):
        self.init_test_case()
        self.expert_count = self.expert_count.astype("int64")

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            capacity = paddle.static.data(
                'capacity', shape=self.capacity.shape, dtype="int64")
            expert_count_tensor = paddle.static.data(
                'ExpertCount', shape=self.expert_count.shape, dtype="int64")
            out = utils._limit_by_capacity(expert_count_tensor, capacity,
                                           self.n_worker)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={
                'capacity': self.capacity,
                'ExpertCount': self.expert_count,
            },
                          fetch_list=out)
        assert np.allclose(self.out, res[0])

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        capacity = paddle.to_tensor(self.capacity)
        expert_count_tensor = paddle.to_tensor(self.expert_count)
        out = utils._limit_by_capacity(expert_count_tensor, capacity,
                                       self.n_worker)
        assert np.allclose(self.out, out.numpy())


if __name__ == '__main__':
    unittest.main()
