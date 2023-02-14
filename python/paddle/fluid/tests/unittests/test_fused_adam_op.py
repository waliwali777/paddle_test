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

import unittest

import numpy as np
from op_test import OpTest

import paddle


def fused_adam_step(inputs, attributes, num):
    '''
    Simulate one step of the fused_adam optimizer
    :param inputs: dict of inputs
    :param attributes: dict of attributes
    :return tuple: tuple of output params, moments1, moments2, beta1_pows, beta2_pows
    '''
    params = inputs['Params']
    grads = inputs['Grads']
    moments1 = inputs['Moments1']
    moments2 = inputs['Moments2']
    lr = inputs['LearningRate']
    beta1_pows = inputs['Beta1Pows']
    beta2_pows = inputs['Beta2Pows']

    params_out = []
    moments1_out = []
    moments2_out = []
    beta1_pows_out = []
    beta2_pows_out = []

    epsilon = attributes['epsilon']

    if 'beta1' in attributes:
        beta1 = attributes['beta1']
    else:
        beta1 = inputs['Beta1Tensor'][0][0]
    if 'beta2' in attributes:
        beta2 = attributes['beta2']
    else:
        beta2 = inputs['Beta2Tensor'][0][0]

    for i in range(num):
        moments1_out.append(beta1 * moments1[i] + (1 - beta1) * grads[i])
        moments2_out.append(
            beta2 * moments2[i] + (1 - beta2) * np.square(grads[i])
        )
        lr_t = lr * np.sqrt(1 - beta2_pows[i]) / (1 - beta1_pows[i])
        params_out.append(
            params[i]
            - lr_t * (moments1_out[i] / (np.sqrt(moments2_out[i]) + epsilon))
        )

    for i in range(num):
        beta1_pows_out.append(
            np.array([beta1_pows[i]]).astype("float32") * beta1
        )
        beta2_pows_out.append(
            np.array([beta2_pows[i]]).astype("float32") * beta2
        )

    return (
        params_out,
        moments1_out,
        moments2_out,
        beta1_pows_out,
        beta2_pows_out,
    )


class TestFusedAdamOp(OpTest):
    def setUp(self):
        '''Test FusedAdam Op with supplied attributes'''
        self.__class__.op_type = "fused_adam"

        num = 10

        inputs_list = [[0] * num] * 6

        learning_rate = 0.004
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 1e-4
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.attrs = {'epsilon': epsilon, 'beta1': beta1, 'beta2': beta2}

        for i in range(num):

            inputs_list[0][i] = np.random.uniform(-1, 1, (102, 105)).astype(
                "float32"
            )
            inputs_list[1][i] = np.random.uniform(-1, 1, (102, 105)).astype(
                "float32"
            )
            inputs_list[2][i] = np.random.uniform(-1, 1, (102, 105)).astype(
                "float32"
            )
            inputs_list[3][i] = np.random.random((102, 105)).astype("float32")
            inputs_list[4][i] = np.array([beta1_pow]).astype("float32")
            inputs_list[5][i] = np.array([beta2_pow]).astype("float32")

        self.inputs = {
            'Params': inputs_list[0],
            'Grads': inputs_list[1],
            'Moments1': inputs_list[2],
            'Moments2': inputs_list[3],
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pows': inputs_list[4],
            'Beta2Pows': inputs_list[5],
        }

        (
            param_out,
            moments1_out,
            moments2_out,
            beta1_pows_out,
            beta2_pows_out,
        ) = fused_adam_step(self.inputs, self.attrs, num)

        self.outputs = {
            'Moments1Out': moments1_out,
            'Moments2Out': moments2_out,
            'ParamsOut': param_out,
            'Beta1PowsOut': beta1_pows_out,
            'Beta2PowsOut': beta2_pows_out,
        }

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
