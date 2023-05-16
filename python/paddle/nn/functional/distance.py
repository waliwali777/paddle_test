#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import _C_ops
from paddle.fluid.framework import in_dygraph_mode

from ...fluid.data_feeder import check_type, check_variable_and_dtype
from ...fluid.layer_helper import LayerHelper

__all__ = []


def pairwise_distance(x, y, p=2.0, epsilon=1e-6, keepdim=False, name=None):
    r"""

    It computes the pairwise distance between two vectors. The
    distance is calculated by p-oreder norm:

    .. math::

        \Vert x \Vert _p = \left( \sum_{i=1}^n \vert x_i \vert ^ p \right) ^ {1/p}.

    Parameters:
        x (Tensor): Tensor, shape is :math:`[N, D]` or :math:`[D]`, where :math:`N`
            is batch size, :math:`D` is the dimension of vector. Available dtype is
            float16, float32, float64.
        y (Tensor): Tensor, shape is :math:`[N, D]` or :math:`[D]`, where :math:`N`
            is batch size, :math:`D` is the dimension of vector. Available dtype is
            float16, float32, float64.
        p (float, optional): The order of norm. Default: :math:`2.0`.
        epsilon (float, optional): Add small value to avoid division by zero.
            Default: :math:`1e-6`.
        keepdim (bool, optional): Whether to reserve the reduced dimension
            in the output Tensor. The result tensor is one dimension less than
            the result of ``|x-y|`` unless :attr:`keepdim` is True. Default: False.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`.
            Generally, no setting is required. Default: None.

    Returns:
        Tensor, the dtype is same as input tensor.

        - If :attr:`keepdim` is True, the output shape is :math:`[N, 1]` or :math:`[1]`,
          depending on whether the input has data shaped as :math:`[N, D]`.
        - If :attr:`keepdim` is False, the output shape is :math:`[N]` or :math:`[]`,
          depending on whether the input has data shaped as :math:`[N, D]`.

    Examples:
        .. code-block:: python

            import paddle
            x = paddle.to_tensor([[1., 3.], [3., 5.]], dtype=paddle.float64)
            y = paddle.to_tensor([[5., 6.], [7., 8.]], dtype=paddle.float64)
            distance = paddle.nn.functional.pairwise_distance(x, y)
            print(distance)
    #       Tensor(shape=[2], dtype=float64, place=Place(gpu:0), stop_gradient=True,
    #              [4.99999860, 4.99999860])

    """
    if in_dygraph_mode():
        sub = _C_ops.subtract(x, y)
        # p_norm op has not uesd epsilon, so change it to the following.
        if epsilon != 0.0:
            epsilon = paddle.fluid.dygraph.base.to_variable(
                [epsilon], dtype=sub.dtype
            )
            sub = _C_ops.add(sub, epsilon)
        return _C_ops.p_norm(sub, p, -1, 0.0, keepdim, False)

    else:
        check_type(p, 'porder', (float, int), 'PairwiseDistance')
        check_type(epsilon, 'epsilon', (float), 'PairwiseDistance')
        check_type(keepdim, 'keepdim', (bool), 'PairwiseDistance')

        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64'], 'PairwiseDistance'
        )
        check_variable_and_dtype(
            y, 'y', ['float16', 'float32', 'float64'], 'PairwiseDistance'
        )
        sub = paddle.subtract(x, y)
        if epsilon != 0.0:
            epsilon_var = sub.block.create_var(dtype=sub.dtype)
            epsilon_var = paddle.full(
                shape=[1], fill_value=epsilon, dtype=sub.dtype
            )
            sub = paddle.add(sub, epsilon_var)
        helper = LayerHelper("PairwiseDistance", name=name)
        attrs = {
            'axis': -1,
            'porder': p,
            'keepdim': keepdim,
            'epsilon': 0.0,
        }
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type='p_norm', inputs={'X': sub}, outputs={'Out': out}, attrs=attrs
        )

        return out


def cdist(
    x, y, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary', name=None
):
    r"""

    Computes batched the p-norm distance between each pair of the two collections of row vectors.

    This function is equivalent to `scipy.spatial.distance.cdist(input,'minkowski', p=p)`
        if :math:`p \in (0, \infty)`. When :math:`p = 0` it is equivalent to `scipy.spatial.distance.cdist(input, 'hamming') * M`.
        When :math:`p = \infty`, the closest scipy function is `scipy.spatial.distance.cdist(xn, lambda x, y: np.abs(x - y).max())`.

    Parameters:
        x (Tensor): Tensor, shape is :math:`B \times P \times M`.
        y (Tensor): Tensor, shape is :math:`B \times R \times M`.
        p (float, optional): The value for the p-norm distance to calculate between each vector pair. Default: :math:`2.0`.
        compute_mode (str, optional):
            - 'use_mm_for_euclid_dist_if_necessary' - will use matrix multiplication approach to
                calculate euclidean distance (p = 2) if P > 25 or R > 25
            - 'use_mm_for_euclid_dist' - will always use matrix multiplication approach to
                calculate euclidean distance (p = 2)
            - 'donot_use_mm_for_euclid_dist' - will never use matrix multiplication approach to
                calculate euclidean distance (p = 2) Default: use_mm_for_euclid_dist_if_necessary.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`.
            Generally, no setting is required. Default: None.

    Returns:
        Tensor, the dtype is same as input tensor.

        If x1 has shape :math:`B \times P \times M` and x2 has shape :math:`B \times R \times M` then
            the output will have shape :math:`B \times P \times R`.

    Examples:
        .. code-block:: python

            import paddle
            x = paddle.to_tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]], dtype=paddle.float32)
            y = paddle.to_tensor([[-2.1763, -0.4713], [-0.6986,  1.3702]], dtype=paddle.float32)
            distance = paddle.nn.functional.cdist(x, y)
            print(distance)
    #       Tensor(shape=[3, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
    #              [[3.1193, 2.0959], [2.7138, 3.8322], [2.2830, 0.3791]])

    """

    check_type(x, 'x', ('float32', 'float64'), 'cdist')
    check_type(y, 'y', ('float32', 'float64'), 'cdist')
    check_type(p, 'p', (float), 'cdist')
    check_type(compute_mode, 'compute_mode', (str), 'cdist')

    x_shape = list(x.shape)
    assert len(x_shape) >= 2, (
        "The x must be at least 2-dimensional, "
        "But received Input x's dimensional is %s.\n" % len(x_shape)
    )
    y_shape = list(y.shape)
    assert len(y_shape) >= 2, (
        "The y must be at least 2-dimensional, "
        "But received Input y's dimensional is %s.\n" % len(y_shape)
    )

    if in_dygraph_mode():
        return _C_ops.cdist(x, y, p, compute_mode)

    helper = LayerHelper("cdist", name=name)
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='cdist',
        inputs={'x': x, 'y': y},
        outputs={'out': out},
        attrs={'p': p, 'compute_mode': compute_mode},
    )
    return out
