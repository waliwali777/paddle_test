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

import unittest

from legacy_test.test_parallel_dygraph_dataparallel import TestMultipleGpus


class TestClusterPartition(unittest.TestCase):
    def test_cluster_partition(self):
        clusters = [(5, 8), (1, 8), (4, 8), (16, 8), (2, 8), (3, 8)]
        from paddle.distributed.auto_parallel.tuner.rule_based_tuner import (
            ClusterPartitionUtil,
        )

        device_meshes = []
        for cluster in clusters:
            n = cluster[0]
            m = cluster[1]
            device_mesh = ClusterPartitionUtil.partition_cluster(n, m)
            device_meshes.append(device_mesh)


class TestHybridParallel(TestMultipleGpus):
    # check sharding logic as well as the accuracy with single mode
    def test_hybrid_parallel_sharding_logic(self):
        self.run_mnist_2gpu('hybrid_parallel_sharding_model.py')

    def test_hybrid_parallel_sharding_tensor_fusion(self):
        self.run_mnist_2gpu('hybrid_parallel_sharding_model_with_fusion.py')

    def test_hybrid_parallel_sharding_state_dict(self):
        self.run_mnist_2gpu('hybrid_parallel_sharding_state_dict.py')


if __name__ == "__main__":
    unittest.main()
