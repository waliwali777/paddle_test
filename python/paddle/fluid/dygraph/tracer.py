# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import six

from collections import defaultdict
from paddle.fluid import core
from paddle.fluid import framework

__all__ = ['Tracer']


def release_op(op):
    del framework._dygraph_tracer()._ops[op._trace_id].inputs
    del framework._dygraph_tracer()._ops[op._trace_id].outputs
    del framework._dygraph_tracer()._ops[op._trace_id].backward_refs


class Tracer(core.Tracer):
    """
    Python wrapper of dygraph tracer
    """

    def __init__(self, block):
        super(Tracer, self).__init__(block)

        self._ops = defaultdict()
        self._vars = defaultdict()
        self._trace_id = 0
        self._train_mode = True

    def trace_var(self, name, var):
        self._vars[name] = var

    def all_parameters(self):
        return list((item for name, item in six.iteritems(self._vars)
                     if isinstance(item, framework.Parameter)))

    def _clear_ops(self):
        self._ops = defaultdict()
        self._trace_id = 0

    def trace_op(self, op, inputs, outputs, stop_gradient=False):
        trace_backward = self._train_mode and not stop_gradient

        inps = defaultdict(list)
        for k, vars in six.iteritems(inputs):
            if isinstance(vars, framework.Variable):
                inps[k].append(vars._ivar)
            elif isinstance(vars, list) or isinstance(vars, tuple):
                for var in vars:
                    inps[k].append(var._ivar)

        outs = defaultdict(list)
        for k, vars in six.iteritems(outputs):
            if isinstance(vars, framework.Variable):
                outs[k].append(vars._ivar)
            elif isinstance(vars, list) or isinstance(vars, tuple):
                for var in vars:
                    outs[k].append(var._ivar)

        self.trace(op._op_type, inps, outs, op.attrs,
                   framework._current_expected_place(), trace_backward)

    def train_mode(self):
        self._train_mode = True

    def eval_mode(self):
        self._train_mode = False
