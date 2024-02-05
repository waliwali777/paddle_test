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

from test_parallel_dygraph_dataparallel import TestMultipleAccelerators


class TestDygraphFleetApi(TestMultipleAccelerators):
    def test_dygraph_fleet_api(self):
        self.run_mnist_2accelerators('dygraph_fleet_api.py')


if __name__ == "__main__":
    unittest.main()
