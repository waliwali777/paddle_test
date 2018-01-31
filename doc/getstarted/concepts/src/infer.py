#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

import paddle.v2 as paddle
import numpy as np

paddle.init(use_gpu=False)
x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(2))
y_predict = paddle.layer.fc(input=x, size=1, act=paddle.activation.Linear())

# loading the model which generated by training
with open('params_pass_90.tar', 'r') as f:
    parameters = paddle.parameters.Parameters.from_tar(f)

# Input multiple sets of data，Output the infer result in a array.
i = [[[1, 2]], [[3, 4]], [[5, 6]]]
print paddle.infer(output_layer=y_predict, parameters=parameters, input=i)
# Will print:
# [[ -3.24491572]
#  [ -6.94668722]
#  [-10.64845848]]
