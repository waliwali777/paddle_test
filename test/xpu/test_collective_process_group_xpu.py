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

from xpu.test_parallel_dygraph_dataparallel import TestMultipleXpus

import paddle
from paddle import core


class TestProcessGroup(TestMultipleXpus):
    @unittest.skipIf(
        not core.is_compiled_with_xpu() or paddle.device.xpu.device_count() < 2,
        "run test when having at leaset 2 XPUs.",
    )
    def test_process_group_bkcl(self):
        self.run_mnist_2xpu('process_group_bkcl.py')


if __name__ == "__main__":
    unittest.main()
