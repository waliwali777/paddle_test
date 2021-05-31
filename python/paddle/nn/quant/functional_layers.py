#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from ...fluid.dygraph import layers
from ...tensor import math, manipulation

__all__ = []


class FloatFunctionalLayer(layers.Layer):
    def __init__(self):
        super(FloatFunctionalLayer, self).__init__()


class add(FloatFunctionalLayer):
    def __init__(self):
        super(add, self).__init__()

    def forward(self, x, y, name=None):
        return math.add(x, y, name)


class subtract(FloatFunctionalLayer):
    def __init__(self):
        super(subtract, self).__init__()

    def forward(self, x, y, name=None):
        return math.subtract(x, y, name)


class multiply(FloatFunctionalLayer):
    def __init__(self):
        super(multiply, self).__init__()

    def forward(self, x, y, name=None):
        return math.multiply(x, y, name)


class divide(FloatFunctionalLayer):
    def __init__(self):
        super(divide, self).__init__()

    def forward(self, x, y, name=None):
        return math.divide(x, y, name)


class reshape(FloatFunctionalLayer):
    def __init__(self):
        super(reshape, self).__init__()

    def forward(self, x, shape, name=None):
        return manipulation.reshape(x, shape, name)


class tranpose(FloatFunctionalLayer):
    def __init__(self, name=None):
        super(tranpose, self).__init__()
        self._name = name

    def forward(self, inputs):
        assert len(inputs) == 2, "The inputs should be (x, perm)"
        return manipulation.tranpose(inputs[0], inputs[1], self._name)


class concat(FloatFunctionalLayer):
    def __init__(self, name=None):
        super(concat, self).__init__()
        self._name = name

    def forward(self, inputs):
        assert len(inputs) == 2, "The inputs should be (x, axis)"
        return manipulation.concat(inputs[0], inputs[1], self._name)


class flatten(FloatFunctionalLayer):
    def __init__(self, name=None):
        super(flatten, self).__init__()
        self._name = name

    def forward(self, inputs):
        assert len(
            inputs) == 3, "The inputs should be (x, start_axis, stop_axis)"
        return manipulation.flatten(inputs[0], inputs[1], inputs[2], self._name)
