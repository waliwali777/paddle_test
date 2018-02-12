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

SIGMOID_THRESHOLD_MIN = -40.0
SIGMOID_THRESHOLD_MAX = 13.0
EXP_MAX_INPUT = 40.0


def identity(x):
    return x


def sigmoid(x):
    y = np.copy(x)
    y[x < SIGMOID_THRESHOLD_MIN] = SIGMOID_THRESHOLD_MIN
    y[x > SIGMOID_THRESHOLD_MAX] = SIGMOID_THRESHOLD_MAX
    return 1. / (1. + np.exp(-y))


def tanh(x):
    y = -2. * x
    y[y > EXP_MAX_INPUT] = EXP_MAX_INPUT
    return (2. / (1. + np.exp(y))) - 1.


def relu(x):
    return np.maximum(x, 0)


ACTIVATION = {
    'identity': identity,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu
}


def lstm(
        input,  # T x 4D
        lod,  # 1 x N
        h0=None,  # N x D
        c0=None,  # N x D
        w_h=None,  # D x 4D
        w_b=None,  # 1 x 4D
        w_c=None,  # 1 x 3D
        is_reverse=False,
        act_gate=None,
        act_cell=None,
        act_cand=None):
    def _step(x, w_h, w_c, h_pre, c_pre, act_gate, act_cell, act_cand):
        g = np.dot(h_pre, w_h)  # 1 x 4D
        g = g + x
        g = np.reshape(g, (1, g.size))
        c, g_i, g_f, g_o = np.split(g, 4, axis=1)
        if w_c is None:
            g_i = act_gate(g_i)  # 1 x D
            g_f = act_gate(g_f)  # 1 x D
        else:
            w_ic, w_fc, w_oc = np.split(w_c, 3, axis=1)
            g_i = act_gate(g_i + w_ic * c_pre)  # 1 x D
            g_f = act_gate(g_f + w_fc * c_pre)  # 1 x D
        c = g_f * c_pre + g_i * act_cand(c)  # 1 x D

        if w_c is None:
            g_o = act_gate(g_o)  # 1 x D
        else:
            _, _, w_oc = np.split(w_c, 3, axis=1)
            g_o = act_gate(g_o + w_oc * c)  # 1 x D
        h = g_o * act_cell(c)
        return h, c

    def _reverse(x, lod):
        y = np.zeros_like(x)
        for i in range(len(lod) - 1):
            b, e = lod[i], lod[i + 1]
            y[b:e, :] = np.flip(x[b:e, :], 0)
        return y

    offset = lod[0]
    batch_size = len(offset) - 1
    hidden = []
    cell = []
    input = _reverse(input, offset) if is_reverse else input
    if w_b is not None:
        input = input + np.tile(w_b, (offset[-1], 1))
    for i in range(batch_size):
        # compute one sequence
        seq_len = offset[i + 1] - offset[i]
        x = input[offset[i]:offset[i + 1], :]
        h_pre = h0[i]  # 1 x D
        c_pre = c0[i]  # 1 x D
        for j in range(seq_len):
            # compute one step
            h_pre, c_pre = _step(x[j], w_h, w_c, h_pre, c_pre, act_gate,
                                 act_cell, act_cand)
            hidden.append(h_pre.flatten())
            cell.append(c_pre.flatten())

    hidden = np.array(hidden).astype('float64')
    cell = np.array(cell).astype('float64')

    hidden = _reverse(hidden, offset) if is_reverse else hidden
    cell = _reverse(cell, offset) if is_reverse else cell

    assert hidden.shape == (input.shape[0], input.shape[1] / 4)
    assert cell.shape == (input.shape[0], input.shape[1] / 4)
    return hidden, cell


class TestLstmOp(OpTest):
    def set_argument(self):
        self.lod = [[0, 2, 5, 7]]
        self.D = 16

        self.act_gate = 'sigmoid'
        self.act_cell = 'tanh'
        self.act_cand = 'tanh'

        self.has_initial_state = False
        self.is_reverse = False
        self.use_peepholes = True

    def setUp(self):
        self.set_argument()
        self.op_type = 'lstm'

        T = self.lod[0][-1]
        N = len(self.lod[0]) - 1

        x = np.random.normal(size=(T, 4 * self.D)).astype('float64')
        if self.has_initial_state:
            h0 = np.random.normal(size=(N, self.D)).astype('float64')
            c0 = np.random.normal(size=(N, self.D)).astype('float64')
        else:
            h0 = np.zeros((N, self.D)).astype('float64')
            c0 = np.zeros((N, self.D)).astype('float64')
        w = np.random.normal(size=(self.D, 4 * self.D)).astype('float64')
        if self.use_peepholes:
            b = np.random.normal(size=(1, 7 * self.D)).astype('float64')
        else:
            b = np.random.normal(size=(1, 4 * self.D)).astype('float64')

        w_b = b[:, 0:4 * self.D]
        w_c = b[:, 4 * self.D:] if self.use_peepholes else None
        h, c = lstm(x, self.lod, h0, c0, w, w_b, w_c, self.is_reverse,
                    ACTIVATION[self.act_gate], ACTIVATION[self.act_cell],
                    ACTIVATION[self.act_cand])

        self.inputs = {'Input': (x, self.lod), 'Weight': w}

        self.inputs['Bias'] = b

        if self.has_initial_state:
            self.inputs['H0'] = h0
            self.inputs['C0'] = c0

        self.outputs = {
            'Hidden': (h, self.lod),
            'Cell': (c, self.lod),
        }
        self.attrs = {
            'use_peepholes': self.use_peepholes,
            'is_reverse': self.is_reverse,
            'gate_activation': self.act_gate,
            'cell_activation': self.act_cell,
            'candidate_activation': self.act_cand
        }

    def test_check_output(self):
        self.check_output(atol=1e-8)

    def test_check_grad(self):
        # TODO(qingqing) remove folowing lines after the check_grad is refined.
        N = len(self.lod[0]) - 1
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(
            ['Input', 'Weight', 'Bias'], ['Hidden'], max_relative_error=5e-4)


class TestLstmOpHasInitial(TestLstmOp):
    def set_argument(self):
        self.lod = [[0, 2, 5, 7]]
        self.D = 16

        self.act_gate = 'sigmoid'
        self.act_cell = 'tanh'
        self.act_cand = 'tanh'

        self.has_initial_state = True
        self.is_reverse = True
        self.use_peepholes = True

    def test_check_grad(self):
        # TODO(qingqing) remove folowing lines after the check_grad is refined.
        N = len(self.lod[0]) - 1
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(
            ['Input', 'Weight', 'Bias', 'H0', 'C0'], ['Hidden'],
            max_relative_error=5e-4)

    def test_check_grad_ingore_bias(self):
        N = len(self.lod[0]) - 1
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(
            ['Input', 'Weight'], ['Hidden'],
            max_relative_error=5e-4,
            no_grad_set=set('Bias'))

    def test_check_grad_ingore_weight(self):
        N = len(self.lod[0]) - 1
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(
            ['Input', 'Bias'], ['Hidden'],
            max_relative_error=5e-4,
            no_grad_set=set('Weight'))

    def test_check_grad_ingore_input(self):
        N = len(self.lod[0]) - 1
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(
            ['Weight', 'Bias'], ['Hidden'],
            max_relative_error=5e-4,
            no_grad_set=set('Input'))

    def test_check_grad_ingore_h0(self):
        N = len(self.lod[0]) - 1
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(
            ['Input', 'Weight', 'Bias', 'C0'], ['Hidden'],
            max_relative_error=5e-4,
            no_grad_set=set('H0'))

    def test_check_grad_ingore_c0(self):
        N = len(self.lod[0]) - 1
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(
            ['Input', 'Weight', 'Bias', 'H0'], ['Hidden'],
            max_relative_error=5e-4,
            no_grad_set=set('C0'))


class TestLstmOpRerverse(TestLstmOp):
    def set_argument(self):
        self.lod = [[0, 2, 5, 7]]
        self.D = 16

        self.act_gate = 'sigmoid'
        self.act_cell = 'tanh'
        self.act_cand = 'tanh'

        self.has_initial_state = False
        self.is_reverse = True
        self.use_peepholes = True


class TestLstmOpNotUsePeepholes(TestLstmOp):
    def set_argument(self):
        self.lod = [[0, 2, 5, 7]]
        self.D = 16

        self.act_gate = 'sigmoid'
        self.act_cell = 'tanh'
        self.act_cand = 'tanh'

        self.has_initial_state = False
        self.is_reverse = True
        self.use_peepholes = False


if __name__ == '__main__':
    unittest.main()
