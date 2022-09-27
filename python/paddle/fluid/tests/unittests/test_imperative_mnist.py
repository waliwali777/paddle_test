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

import contextlib
import unittest
import numpy as np
import six

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
from paddle.fluid.dygraph.base import to_variable
from test_imperative_base import new_program_scope
from utils import DyGraphProgramDescTracerTestHelper, is_equal_program
from paddle.fluid.framework import _test_eager_guard, _in_legacy_dygraph


class SimpleImgConvPool(fluid.dygraph.Layer):

    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 pool_size,
                 pool_stride,
                 pool_padding=0,
                 pool_type='max',
                 global_pooling=False,
                 conv_stride=1,
                 conv_padding=0,
                 conv_dilation=1,
                 conv_groups=1,
                 act=None,
                 use_cudnn=False,
                 param_attr=None,
                 bias_attr=None):
        super(SimpleImgConvPool, self).__init__()

        self._conv2d = Conv2D(num_channels=num_channels,
                              num_filters=num_filters,
                              filter_size=filter_size,
                              stride=conv_stride,
                              padding=conv_padding,
                              dilation=conv_dilation,
                              groups=conv_groups,
                              param_attr=None,
                              bias_attr=None,
                              use_cudnn=use_cudnn)

        self._pool2d = Pool2D(pool_size=pool_size,
                              pool_type=pool_type,
                              pool_stride=pool_stride,
                              pool_padding=pool_padding,
                              global_pooling=global_pooling,
                              use_cudnn=use_cudnn)

    def forward(self, inputs):
        x = self._conv2d(inputs)
        x = self._pool2d(x)
        return x


class MNIST(fluid.dygraph.Layer):

    def __init__(self):
        super(MNIST, self).__init__()

        self._simple_img_conv_pool_1 = SimpleImgConvPool(1,
                                                         20,
                                                         5,
                                                         2,
                                                         2,
                                                         act="relu")

        self._simple_img_conv_pool_2 = SimpleImgConvPool(20,
                                                         50,
                                                         5,
                                                         2,
                                                         2,
                                                         act="relu")

        self.pool_2_shape = 50 * 4 * 4
        SIZE = 10
        scale = (2.0 / (self.pool_2_shape**2 * SIZE))**0.5
        self._fc = Linear(self.pool_2_shape,
                          10,
                          param_attr=fluid.param_attr.ParamAttr(
                              initializer=fluid.initializer.NormalInitializer(
                                  loc=0.0, scale=scale)),
                          act="softmax")

    def forward(self, inputs):
        x = self._simple_img_conv_pool_1(inputs)
        x = self._simple_img_conv_pool_2(x)
        x = fluid.layers.reshape(x, shape=[-1, self.pool_2_shape])
        x = self._fc(x)
        return x


class TestImperativeMnist(unittest.TestCase):

    def reader_decorator(self, reader):

        def _reader_imple():
            for item in reader():
                image = np.array(item[0]).reshape(1, 28, 28)
                label = np.array(item[1]).astype('int64').reshape(1)
                yield image, label

        return _reader_imple

    def func_test_mnist_float32(self):
        seed = 90
        epoch_num = 1
        batch_size = 128
        batch_num = 50

        traced_layer = None

        with fluid.dygraph.guard():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed

            mnist = MNIST()
            sgd = SGDOptimizer(learning_rate=1e-3,
                               parameter_list=mnist.parameters())

            batch_py_reader = fluid.io.PyReader(capacity=1)
            batch_py_reader.decorate_sample_list_generator(
                paddle.batch(self.reader_decorator(
                    paddle.dataset.mnist.train()),
                             batch_size=batch_size,
                             drop_last=True),
                places=fluid.CPUPlace())

            mnist.train()
            dy_param_init_value = {}

            helper = DyGraphProgramDescTracerTestHelper(self)
            program = None
            for epoch in range(epoch_num):
                for batch_id, data in enumerate(batch_py_reader()):
                    if batch_id >= batch_num:
                        break
                    img = data[0]
                    dy_x_data = img.numpy()
                    label = data[1]
                    label.stop_gradient = True

                    if batch_id % 10 == 0 and _in_legacy_dygraph():
                        cost, traced_layer = paddle.jit.TracedLayer.trace(
                            mnist, inputs=img)
                        if program is not None:
                            self.assertTrue(program, traced_layer.program)
                        program = traced_layer.program
                        traced_layer.save_inference_model(
                            './infer_imperative_mnist')
                    else:
                        cost = mnist(img)

                    if traced_layer is not None:
                        cost_static = traced_layer([img])
                        helper.assertEachVar(cost, cost_static)

                    loss = fluid.layers.cross_entropy(cost, label)
                    avg_loss = paddle.mean(loss)

                    dy_out = avg_loss.numpy()

                    if epoch == 0 and batch_id == 0:
                        for param in mnist.parameters():
                            dy_param_init_value[param.name] = param.numpy()

                    avg_loss.backward()
                    sgd.minimize(avg_loss)
                    mnist.clear_gradients()

                    dy_param_value = {}
                    for param in mnist.parameters():
                        dy_param_value[param.name] = param.numpy()

        with new_program_scope():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed

            exe = fluid.Executor(fluid.CPUPlace(
            ) if not core.is_compiled_with_cuda() else fluid.CUDAPlace(0))

            mnist = MNIST()
            sgd = SGDOptimizer(learning_rate=1e-3)
            train_reader = paddle.batch(paddle.dataset.mnist.train(),
                                        batch_size=batch_size,
                                        drop_last=True)

            img = fluid.layers.data(name='pixel',
                                    shape=[1, 28, 28],
                                    dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            cost = mnist(img)
            loss = fluid.layers.cross_entropy(cost, label)
            avg_loss = paddle.mean(loss)
            sgd.minimize(avg_loss)

            # initialize params and fetch them
            static_param_init_value = {}
            static_param_name_list = []
            for param in mnist.parameters():
                static_param_name_list.append(param.name)

            out = exe.run(fluid.default_startup_program(),
                          fetch_list=static_param_name_list)

            for i in range(len(static_param_name_list)):
                static_param_init_value[static_param_name_list[i]] = out[i]

            for epoch in range(epoch_num):
                for batch_id, data in enumerate(train_reader()):
                    if batch_id >= batch_num:
                        break
                    static_x_data = np.array([
                        x[0].reshape(1, 28, 28) for x in data
                    ]).astype('float32')
                    y_data = np.array([x[1]
                                       for x in data]).astype('int64').reshape(
                                           [batch_size, 1])

                    fetch_list = [avg_loss.name]
                    fetch_list.extend(static_param_name_list)

                    if traced_layer is not None:
                        traced_layer([static_x_data])

                    out = exe.run(fluid.default_main_program(),
                                  feed={
                                      "pixel": static_x_data,
                                      "label": y_data
                                  },
                                  fetch_list=fetch_list)

                    static_param_value = {}
                    static_out = out[0]
                    for i in range(1, len(out)):
                        static_param_value[static_param_name_list[i -
                                                                  1]] = out[i]

        np.testing.assert_allclose(dy_x_data.all(),
                                   static_x_data.all(),
                                   rtol=1e-05)

        for key, value in six.iteritems(static_param_init_value):
            np.testing.assert_allclose(value,
                                       dy_param_init_value[key],
                                       rtol=1e-05)

        np.testing.assert_allclose(static_out, dy_out, rtol=1e-05)

        for key, value in six.iteritems(static_param_value):
            np.testing.assert_allclose(value,
                                       dy_param_value[key],
                                       rtol=1e-05,
                                       atol=1e-05)

    def test_mnist_float32(self):
        with _test_eager_guard():
            self.func_test_mnist_float32()
        self.func_test_mnist_float32()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
