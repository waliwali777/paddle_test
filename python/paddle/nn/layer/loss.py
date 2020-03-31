#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
# TODO: define loss functions of neural network
__all__ = [
    #            'NCELoss',
    'CrossEntropyLoss',
    #            'MSELoss',
    #            'L1Loss',
    #            'NLLLoss',
    #            'BCELoss'
]


class CrossEntropyLoss(fluid.dygraph.Layer):
    """
    This operator implements the cross entropy loss function.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument `weight` should be a 1D Variable assigning
    weight to each of the classes.

    For predictions label, and target label, the loss is calculated as follows.
    .. math::

        loss_j =  -\\text{input[class]} +
        \\log\\left(\\sum_{i=0}^{K}\\exp(\\text{input}_i)\\right), j = 1,..., K

    If weight is not `None`:
    .. math::

        loss_j =  \\text{weight[class]}(-\\text{input[class]} +
        \\log\\left(\\sum_{i=0}^{K}\\exp(\\text{input}_i)\\right)), j = 1,..., K

    Parameters:
        input (Variable): Input tensor, the data type is float32,
            float64, int32, int64.
        label (Variable): Label tensor, the data type is float32,
            float64, int32, int64.
        weight (Variable, optional): Weight tensor, a manual rescaling weight given
            to each class. It has the same dimensions as class number and the data type
            is float32, float64, int32, int64. Default is ``'None'``.
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`size_average` is ``'sum'``, the reduced sum loss is returned.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned.
            Default is ``'mean'``.
    Returns:
        The tensor variable storing the cross_entropy_loss of input and label.
    Return type: Variable.
    Examples:
        .. code-block:: python

            # declarative mode
            import paddle
            import paddle.fluid as fluid
            import numpy as np

            input = fluid.layers.data(name='input', shape=[3,5], dtype='float32')
            label = fluid.layers.data(name='label', shape=[3,1], dtype='int64')
            weight = fluid.layers.data(name='weight', shape=[5], dtype='float32')
            ce_loss = paddle.nn.loss.CrossEntropyLoss(weight=weight, reduction='mean')
            output = ce_loss(input,label)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            input_data = np.random.random([3,5]).astype("float32")
            label_data = np.array([[1], [3], [4]]).astype("int64")
            weight_data = np.random.random([5]).astype("float32")
            output = exe.run(fluid.default_main_program(),
                    feed={"input":input_data, "label":label_data,"weight":weight_data},
                    fetch_list=[output],
                    return_numpy=True)
            print(output)

            # imperative mode
            import paddle.fluid.dygraph as dg
            with dg.guard(place) as g:
                input = dg.to_variable(input_data)
                label = dg.to_variable(label_data)
                weight = dg.to_variable(weight_data)
                ce_loss = paddle.nn.loss.CrossEntropyLoss(weight=weight, reduction='mean')
                output = ce_loss(input,label)
                print(output.numpy())
    """

    def __init__(self, weight=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, label):
        fluid.data_feeder.check_variable_and_dtype(
            input, 'input', ['float32', 'float64', 'int32', 'int64'],
            'cross_entropy_loss')
        fluid.data_feeder.check_variable_and_dtype(
            label, 'label', ['float32', 'float64', 'int32', 'int64'],
            'cross_entropy_loss')

        if self.reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in cross_entropy_loss should be 'sum', 'mean' or 'none',"
                " but received %s, which is not allowed." % self.reduction)

        softmax_out = fluid.layers.softmax(input)
        if self.weight is not None:
            if isinstance(self.weight, fluid.framework.Variable):
                softmax_out = fluid.layers.elementwise_pow(
                    softmax_out, self.weight, axis=-1)
            else:
                raise ValueError(
                    "The weight' is not a Variable, please convert to Variable.")

        out = fluid.layers.cross_entropy(softmax_out, label)

        if self.reduction == 'sum':
            return fluid.layers.reduce_sum(out)
        elif self.reduction == 'mean':
            return fluid.layers.reduce_mean(out)
        else:
            return out
