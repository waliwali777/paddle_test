# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from ...static import Variable
from ...fluid.layer_helper import LayerHelper
from ...fluid.data_feeder import check_variable_and_dtype
from paddle import _C_ops, _legacy_C_ops
from ...fluid.framework import _in_legacy_dygraph, in_dygraph_mode

__all__ = []


def one_hot(x, num_classes, name=None):
    """

    The operator converts each id in the input 'x' to an one-hot vector with a
    num_classes length. The value in the vector dimension corresponding to the id
    is 1, and the value in the remaining dimension is 0.

    The shape of output Tensor is generated by appending num_classes dimension
    behind the last dimension of the 'x' shape.

    .. code-block:: text

        Example 1:

        input:
            x.shape = [4]
            x.data = [1, 1, 3, 0]
            num_classes = 4

        output:
            Out.shape = [4, 4]
            Out.data = [[0., 1., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 0., 1.],
                        [1., 0., 0., 0.]]

        Example 2:

        input:
            x.shape = [4]
            x.data = [1, 1, 5, 0]
            num_classes = 4

        output: Throw an exception for Illegal value
            The second dimension in X is 5, which is greater than num_classes,
            so it throws an exception.


    Args:
        x(Tensor): Tensor with shape :math:`[N_1, N_2, ..., N_k]` ,
            which contains at least one dimension. The data type is int32 or int64.
        num_classes(int): An integer defining the num_classes of the one hot dimension. If input 'x'
            is word id, num_classes is generally the dictionary size.

    Returns:
        Tensor: The one-hot representations of 'x'. A Tensor with type float32.

    Examples:
        .. code-block:: python

            import paddle
            # Correspond to the first example above, where label.shape is 4 and one_hot_label.shape is [4, 4].
            label = paddle.to_tensor([1, 1, 3, 0], dtype='int64')
            # label.shape = [4]
            one_hot_label = paddle.nn.functional.one_hot(label, num_classes=4)
            # one_hot_label.shape = [4, 4]
            # one_hot_label = [[0., 1., 0., 0.],
            #                  [0., 1., 0., 0.],
            #                  [0., 0., 0., 1.],
            #                  [1., 0., 0., 0.]]

    """

    if in_dygraph_mode():
        return _C_ops.one_hot(x, num_classes)
    else:
        if _in_legacy_dygraph():
            return _legacy_C_ops.one_hot_v2(x, 'depth', num_classes,
                                            'allow_out_of_range', False)
        else:
            check_variable_and_dtype(x, 'input', ['int32', 'int64'],
                                     'one_hot_v2')
            helper = LayerHelper("one_hot_v2", **locals())

            one_hot_out = helper.create_variable_for_type_inference(
                dtype='float32')
            if not isinstance(num_classes, Variable):
                # user attribute
                inputs = {'X': x}
                attrs = {'depth': num_classes, 'allow_out_of_range': False}
            else:
                num_classes.stop_gradient = True
                inputs = {'X': x, 'depth_tensor': num_classes}
                attrs = {'allow_out_of_range': False}
            helper.append_op(type="one_hot_v2",
                             inputs=inputs,
                             attrs=attrs,
                             outputs={'Out': one_hot_out},
                             stop_gradient=True)
            return one_hot_out


def embedding(x, weight, padding_idx=None, sparse=False, name=None):
    r"""
    Used to lookup embeddings vector of ids provided by :attr:`x` .

    The shape of output Tensor is generated by appending the last dimension of the input Tensor shape
    with embedding size.

    Note:
        The id in :attr:`x` must satisfy :math:`0 =< id < weight.shape[0]` ,
        otherwise the program will throw an exception and exit.

    .. code-block:: text

            x is a Tensor.
                padding_idx = -1
                x.data = [[1, 3], [2, 4], [4, 127]]
                x.shape = [3, 2]
                weight.shape = [128, 16]
            output is a Tensor:
                out.shape = [3, 2, 16]
                out.data = [[[0.129435295, 0.244512452, ..., 0.436322452],
                            [0.345421456, 0.524563927, ..., 0.144534654]],
                            [[0.345249859, 0.124939536, ..., 0.194353745],
                            [0.945345345, 0.435394634, ..., 0.435345365]],
                            [[0.945345345, 0.435394634, ..., 0.435345365],
                            [0.0,         0.0,         ..., 0.0        ]]]  # padding data

            The input padding_idx is less than 0, it is automatically converted to padding_idx = -1 + 128 = 127
            It will pad all-zero data when id is 127.

    Args:
        x(Tensor): A Tensor with type int32/int64, which contains the id information. The value of the input id should
            satisfy :math:`0<= id < weight.shape[0]` .
        weight (Tensor): The weight. A Tensor with shape of lookup table parameter. It should have two elements which
            indicates the size of the dictionary of embeddings and the size of each embedding vector respectively.
        sparse(bool, optional): The flag indicating whether to use sparse update. This parameter only
            affects the performance of the backwards gradient update. It is recommended to set
            True because sparse update is faster. But some optimizers does not support sparse update,
            such as :ref:`api_paddle_optimizer_adadelta_Adadelta` , :ref:`api_paddle_optimizer_adamax_Adamax` , :ref:`api_paddle_optimizer_lamb_Lamb`.
            In these cases, sparse must be False. Default: False.
        padding_idx(int|long|None, optional): padding_idx needs to be in the interval [-weight.shape[0], weight.shape[0]).
            If :math:`padding\_idx < 0`, the :math:`padding\_idx` will automatically be converted
            to :math:`weight.shape[0] + padding\_idx` . It will output all-zero padding data whenever lookup
            encounters :math:`padding\_idx` in id. And the padding data will not be updated while training.
            If set None, it makes no effect to output. Default: None.
        name(str|None, optional): For detailed information, please refer
           to :ref:`api_guide_Name`. Usually name is no need to set and
           None by default.

    Returns:
        Tensor: Embedding Tensor  mapped by x. The data type is the same as :attr:`weight`.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.nn as nn

            x0 = paddle.arange(3, 6).reshape((3, 1)).astype(paddle.int64)
            w0 = paddle.full(shape=(10, 3), fill_value=2).astype(paddle.float32)

            # x.data = [[3], [4], [5]]
            # x.shape = [3, 1]
            x = paddle.to_tensor(x0, stop_gradient=False)

            # w.data = [[2. 2. 2.] ... [2. 2. 2.]]
            # w.shape = [10, 3]
            w = paddle.to_tensor(w0, stop_gradient=False)

            # emb.data = [[[2., 2., 2.]], [[2., 2., 2.]], [[2., 2., 2.]]]
            # emb.shape = [3, 1, 3]
            emb = nn.functional.embedding(
                    x=x, weight=w, sparse=True, name="embedding")

    """
    padding_idx = -1 if padding_idx is None else padding_idx if padding_idx >= 0 else (
        weight.shape[0] + padding_idx)

    if padding_idx >= weight.shape[0] or padding_idx < -weight.shape[0]:
        raise ValueError("padding_idx must be within [-{}, {})".format(
            weight.shape[0], weight.shape[0]))

    if in_dygraph_mode():
        return _C_ops.embedding(x, weight, padding_idx, sparse)
    elif _in_legacy_dygraph():
        return _legacy_C_ops.lookup_table_v2(weight, x, 'is_sparse', sparse,
                                             'is_distributed', False,
                                             'remote_prefetch', False,
                                             'padding_idx', padding_idx)
    else:
        helper = LayerHelper('embedding', **locals())
        dtype = helper.input_dtype(input_param_name='weight')

        check_variable_and_dtype(x, 'input',
                                 ['uint8', 'int8', 'int16', 'int32', 'int64'],
                                 'embedding')

        is_distributed = False
        remote_prefetch = sparse and (not is_distributed)

        tmp = helper.create_variable_for_type_inference(dtype)

        helper.append_op(type='lookup_table_v2',
                         inputs={
                             'Ids': x,
                             'W': weight
                         },
                         outputs={'Out': tmp},
                         attrs={
                             'is_sparse': sparse,
                             'is_distributed': is_distributed,
                             'remote_prefetch': remote_prefetch,
                             'padding_idx': padding_idx
                         })
        return tmp
