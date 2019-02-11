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

from __future__ import print_function

import unittest
from paddle.fluid.tests.unittests.test_batch_norm_op import TestBatchNormOpTraining, TestBatchNormOpInference


class TestNGRAPHBatchNormOpTraining(TestBatchNormOpTraining):
    def init_kernel_type(self):
        super(TestNGRAPHBatchNormOpTraining, self).init_kernel_type()


class TestNGRAPHBatchNormOpInference(TestBatchNormOpInference):
    def init_kernel_type(self):
        super(TestNGRAPHBatchNormOpInference, self).init_kernel_type()


class TestNGRAPHBatchNormOpWithReluInference(TestBatchNormOpInference):
    def init_kernel_type(self):
        super(TestNGRAPHBatchNormOpWithReluInference, self).init_kernel_type()


if __name__ == '__main__':
    unittest.main()
