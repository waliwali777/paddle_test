# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
    enable_to_static_guard,
    test_legacy_and_pt_and_pir,
)

import paddle
from paddle.jit.api import to_static


@to_static
def func_test_to_static():
    x = paddle.arange(12, dtype=paddle.float32).reshape([2, 3, 2])
    x = x.transpose([1, 0, 2])
    x = x.contiguous()
    assert x.is_contiguous()
    return x


class TestContiguous(Dy2StTestBase):
    @test_legacy_and_pt_and_pir
    def test_to_static(self):
        with enable_to_static_guard(True):
            static_result = func_test_to_static()
        with enable_to_static_guard(False):
            dygraph_result = func_test_to_static()
        np.testing.assert_allclose(
            static_result,
            dygraph_result,
            err_msg=f'static_result: {static_result} \n dygraph_result: {dygraph_result}',
        )


if __name__ == '__main__':
    unittest.main()
