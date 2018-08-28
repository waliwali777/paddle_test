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

import copy
import unittest

import numpy as np
from paddle.fluid import core
from paddle.fluid.op import Operator

from op_test import OpTest


class TestAdamOp1(OpTest):
    def setUp(self):
        '''Test Adam Op with supplied attributes
        '''
        self.op_type = "adam"
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")

        learning_rate = 0.004
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 1e-4
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment1': moment1,
            'Moment2': moment2,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32"),
            'Beta2Pow': np.array([beta2_pow]).astype("float32")
        }

        self.attrs = {'epsilon': epsilon, 'beta1': beta1, 'beta2': beta2}

        param_out, moment1_out, \
            moment2_out = adam_step(self.inputs, self.attrs)

        self.outputs = {
            'Moment1Out': moment1_out,
            'Moment2Out': moment2_out,
            'ParamOut': param_out
        }

    def test_check_output(self):
        self.check_output()


class TestAdamOp2(OpTest):
    def setUp(self):
        '''Test Adam Op with supplied attributes
        '''
        self.op_type = "adam"
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")

        learning_rate = 0.001
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment1': moment1,
            'Moment2': moment2,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32"),
            'Beta2Pow': np.array([beta2_pow]).astype("float32")
        }

        attributes = {'epsilon': epsilon, 'beta1': beta1, 'beta2': beta2}

        param_out, moment1_out, \
            moment2_out = adam_step(self.inputs, attributes)

        self.outputs = {
            'Moment1Out': moment1_out,
            'Moment2Out': moment2_out,
            'ParamOut': param_out
        }

    def test_check_output(self):
        self.check_output()


class TestAdamOpMultipleSteps(OpTest):
    def setUp(self):
        '''Test Adam Operator with supplied attributes
        '''
        self.op_type = "adam"
        self.num_steps = 10

        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")

        learning_rate = 0.001
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment1': moment1,
            'Moment2': moment2,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32"),
            'Beta2Pow': np.array([beta2_pow]).astype("float32")
        }

        self.attrs = {'epsilon': epsilon, 'beta1': beta1, 'beta2': beta2}

    def test_check_output(self):
        for _ in range(self.num_steps):
            param_out, moment1_out, \
                moment2_out = adam_step(self.inputs, self.attrs)

            self.outputs = {
                'Moment1Out': moment1_out,
                'Moment2Out': moment2_out,
                'ParamOut': param_out
            }

            # Verify output for this step
            self.check_output()

            # Output of this step becomes input for next step
            self.inputs['Param'] = param_out
            self.inputs['Moment1'] = moment1_out
            self.inputs['Moment2'] = moment2_out

            # Update powers of Beta1 and Beta2 for next time step
            self.inputs['Beta1Pow'] *= self.attrs['beta1']
            self.inputs['Beta2Pow'] *= self.attrs['beta1']

            # Randomize gradient for next step
            self.inputs['Grad'] = np.random.uniform(
                -1, 1, (102, 105)).astype("float32")


def adam_step(inputs, attributes):
    '''
    Simulate one step of the adam optimizer
    :param inputs: dict of inputs
    :param attributes: dict of attributes
    :return tuple: tuple of output param, moment1, moment2,
    beta1 power accumulator and beta2 power accumulator
    '''
    param = inputs['Param']
    grad = inputs['Grad']
    moment1 = inputs['Moment1']
    moment2 = inputs['Moment2']
    lr = inputs['LearningRate']
    beta1_pow = inputs['Beta1Pow']
    beta2_pow = inputs['Beta2Pow']

    beta1 = attributes['beta1']
    beta2 = attributes['beta2']
    epsilon = attributes['epsilon']

    moment1_out = beta1 * moment1 + (1 - beta1) * grad
    moment2_out = beta2 * moment2 + (1 - beta2) * np.square(grad)
    lr_t = lr * np.sqrt(1 - beta2_pow) / (1 - beta1_pow)
    param_out = param - lr_t * (moment1_out / (np.sqrt(moment2_out) + epsilon))
    return param_out, moment1_out, moment2_out


def adam_step_sparse(init_inputs, attributes, height, rows, row_numel, np_grad,
                     steps):
    '''
    Simulate one step of the adam optimizer
    :param inputs: dict of inputs
    :param attributes: dict of attributes
    :return tuple: tuple of output param, moment1, moment2,
    beta1 power accumulator and beta2 power accumulator
    '''
    inputs = copy.deepcopy(init_inputs)

    param = inputs['Param']
    # grad = inputs['Grad']
    moment1 = inputs['Moment1']
    moment2 = inputs['Moment2']
    lr = inputs['LearningRate']
    beta1_pow = inputs['Beta1Pow']
    beta2_pow = inputs['Beta2Pow']

    beta1 = attributes['beta1']
    beta2 = attributes['beta2']
    epsilon = attributes['epsilon']

    for step in range(steps):
        for row_id in range(height):
            grad1 = 0
            grad2 = 0
            if row_id in rows:
                idx = rows.index(row_id)
                grad1 = (1 - beta1) * np_grad[idx]
                grad2 = (1 - beta2) * np.square(np_grad[idx])
            moment1[row_id] = beta1 * moment1[row_id] + grad1
            moment2[row_id] = beta2 * moment2[row_id] + grad2
        lr_t = lr * np.sqrt(1 - beta2_pow) / (1 - beta1_pow)
        param = param - lr_t * (moment1 / (np.sqrt(moment2) + epsilon))
    return param, moment1, moment2


class TestSparseAdamOp(unittest.TestCase):
    def setup(self, scope, place, run_steps):
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 1e-4

        self.height = 10
        self.rows = [0, 4, 7]
        self.row_numel = 12

        self.dense_inputs = {
            "Param": np.full((self.height, self.row_numel),
                             5.0).astype("float32"),
            "Moment1": np.full((self.height, self.row_numel),
                               5.0).astype("float32"),
            "Moment2": np.full((self.height, self.row_numel),
                               5.0).astype("float32"),
            'Beta1Pow': np.array([beta1]).astype("float32"),
            'Beta2Pow': np.array([beta2]).astype("float32"),
            "LearningRate": np.array([2.0]).astype("float32")
        }

        for key, np_array in self.dense_inputs.items():
            var = scope.var(key).get_tensor()
            var.set(np_array, place)

        self.attrs = {'epsilon': epsilon, 'beta1': beta1, 'beta2': beta2}

        # get grad selected rows
        grad_selected_rows = scope.var('Grad').get_selected_rows()
        grad_selected_rows.set_height(self.height)
        grad_selected_rows.set_rows(self.rows)

        self.grad_array = np.ones(
            (len(self.rows), self.row_numel)).astype("float32")
        self.grad_array[0, 0] = 2.0
        self.grad_array[2, 8] = 4.0

        grad_tensor = grad_selected_rows.get_tensor()
        grad_tensor.set(self.grad_array, place)

    def run_python_adam(self, run_steps):
        param_out, mom1, mom2 = adam_step_sparse(
            self.dense_inputs, self.attrs, self.height, self.rows,
            self.row_numel, self.grad_array, run_steps)
        self.outputs = {"Param": param_out, "Moment1": mom1, "Moment2": mom2}

    def run_adam_op(self, scope, place, run_steps):
        adam_op = Operator(
            "adam",
            Param='Param',
            ParamOut='Param',
            Grad='Grad',
            Moment1='Moment1',
            Moment2='Moment2',
            Moment1Out='Moment1',
            Moment2Out='Moment2',
            Beta1Pow='Beta1Pow',
            Beta2Pow='Beta2Pow',
            LearningRate='LearningRate',
            beta1=self.attrs['beta1'],
            beta2=self.attrs['beta2'],
            epsilon=self.attrs['epsilon'])

        for _ in range(run_steps):
            adam_op.run(scope, place)

    def check_with_place(self, place, steps):
        scope = core.Scope()

        self.setup(scope, place, steps)

        self.run_python_adam(steps)

        self.run_adam_op(scope, place, steps)

        for key, np_array in self.outputs.items():
            out_var = scope.var(key).get_tensor()
            actual = np.array(out_var)
            actual = actual.reshape([actual.size])
            np_array = np_array.reshape([np_array.size])
            for idx, row_id in enumerate(self.rows):
                j = 0
                while j < self.row_numel:
                    pos = row_id * self.row_numel + j
                    self.assertLess((actual[pos] - np_array[pos]) / actual[pos],
                                    0.00001)
                    j += 1

    def test_sparse_adam_(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_with_place(place, 20)


if __name__ == "__main__":
    unittest.main()
