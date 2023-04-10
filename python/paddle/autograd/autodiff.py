# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from typing import List, Optional, Sequence, Tuple, Union

import paddle
from paddle.fluid import framework


def as_tensors(xs):
    if isinstance(xs, framework.Variable):
        return (xs,)
    elif isinstance(xs, Sequence):
        return tuple(xs)
    else:
        return xs


class Jacobian:
    r"""
    Computes the Jacobian matrix of a given function.

    If the function has multiple inputs and multiple outputs, during internal
    implementation, all input tensors are concatenated after being flatten,
    the batch dimension is retained, and the output is subject to the same
    processing rules.

    Once the Jacobian ``J`` is constructed, you can use a multidimensional index
    to retrieve the submatrix of ``J``, as same as slicing a Tensor. The
    submatrix is lazily evaluated along row axis, and will be cached once
    evaluated.

    For examples, supposing ``is_batched=True``, you can retrieve the submatrix
    by following methods:

        * J[:], retrieving the full matrix.
        * J[:, :, j], retrieving the partial derivatives w.r.t. the j'th input
          variable.
        * J[:, i, :], retrieving the partial derivatives w.r.t. the i'th output
          variable.
        * J[:, i, j], retrieving the partial derivatives w.r.t. the i'th output
          variable and the j'th input variable.

    Notes:

        Eclipsis index is not supported currently.

    Warning:
        This API is in beta, the signatures could be changed in future version.

    Args:

        ys (Tensor|Sequence[Tensor]): The output derived from xs .
        xs (Tensor|Sequence[Tensor]): The input to the function ``func`` .
        is_batched (bool): If true, the first axis is batch axis. Defaults to
            False.

    Returns:

        Jacobian (Object): A python object retains the Jacobian matrix.

    Examples:

        .. code-block:: python

            import paddle


            def func(x, y):
                return paddle.matmul(x, y)


            x = paddle.to_tensor([[1., 2.], [3., 4.]])
            y = func(x, x)
            J = paddle.incubate.autograd.Jacobian(y, x)
            print(J[:, :])
            # Tensor(shape=[4, 8], dtype=float32, place=Place(gpu:0), stop_gradient=False,
            #        [[1., 3., 0., 0., 1., 0., 2., 0.],
            #         [2., 4., 0., 0., 0., 1., 0., 2.],
            #         [0., 0., 1., 3., 3., 0., 4., 0.],
            #         [0., 0., 2., 4., 0., 3., 0., 4.]])

            print(J[0, :])
            # Tensor(shape=[8], dtype=float32, place=Place(gpu:0), stop_gradient=False,
            #        [1., 3., 0., 0., 1., 0., 2., 0.])
            print(J[:, 0])
            # Tensor(shape=[4], dtype=float32, place=Place(gpu:0), stop_gradient=False,
            #        [1., 2., 0., 0.])

    """

    def __init__(self, ys, xs, is_batched=False):
        if not is_batched:
            self._jacobian = _JacobianNoBatch(ys, xs)
        else:
            self._jacobian = _JacobianBatchFirst(ys, xs)

    @property
    def shape(self):
        """The shape of flattened Jacobian matrix."""
        return self._jacobian.shape

    def __getitem__(self, indexes):
        return self._jacobian[indexes]

    def __getattr__(self, __name: str):
        return getattr(self._jacobian[:], __name)


class Hessian:
    """
    Computes the Hessian matrix  with a given ``func`` with respect to ``xs`` .

    If the function has multiple inputs, during internal implementation,
    all input tensors are concatenated after being flatten, the batch dimension
    is retained.

    The Hessian submatrix is lazily evaluated, and can be retrieved with a
    multidimensional indexes. See details ``Jacobian`` .

    Warning:
        This API is in beta, the signatures could be changed in future version.

    Args:
        func (Callable): A python function that takes a Tensor or a Tensor
            sequence as inputs and returns a Tensor with shape
            ``[batch_size, 1]`` with batch or ``[1]`` without batch.
        xs (Tensor|Sequence(Tensor)): The input Tensor or Tensor sequence of
            the function ``func``.
        is_batched (bool): If true, the first axis is batch axis. Defaults to
            False.

    Returns:

        Hessian (Object): A python object retains the Hessian matrix.


    Examples:

    .. code-block:: python

        import paddle


        def reducer(x):
            return paddle.sum(x * x)


        x = paddle.rand([2, 2])
        h = paddle.incubate.autograd.Hessian(reducer, x)
        print(h[:])
        # Tensor(shape=[4, 4], dtype=float32, place=Place(gpu:0), stop_gradient=False,
        #        [[2., 0., 0., 0.],
        #         [0., 2., 0., 0.],
        #         [0., 0., 2., 0.],
        #         [0., 0., 0., 2.]])
    """

    def __init__(self, ys, xs, is_batched=False):
        _jacobian = Jacobian(ys, xs, is_batched=is_batched)
        _jacobian = _jacobian[:, 0, :] if is_batched else _jacobian[0, :]

        self._hessian = Jacobian(_jacobian, xs, is_batched=is_batched)

    @property
    def shape(self):
        """The shape of flattened Hessian matrix."""
        return self._hessian.shape

    def __getitem__(self, indexes):
        return self._hessian[indexes]

    def __getattr__(self, __name: str):
        return getattr(self._hessian[:], __name)


class _Jacobian:
    """The base class for computing Jacobian matrix.

    ``_Jacobian`` implementes the core logic of multidimensional index and lazy
    evaluation for Jacobian matrix, subclass only need to overwrite following
    methods:

        * ``_lazy_axis()``,  return the axis along which will be lazy
            evaluating.
        * ``_flatten(xs)``, flattens the inputs ``xs``.
        * ``_evaluate(index)``, evaluates one slice along ``_lazy_axis`` .

    Notes:

        Because currently PaddlePaddle only support reverse differentiation by
        ``paddle.grad``, so lazy evaluation is only supported along the row of
        Jacobian matrix, which means that slicing along row will get better
        performance.

    """

    def __init__(self, ys, xs):
        self._xs = xs
        self._ys = ys
        self._flatten_xs = self._flatten(as_tensors(self._xs))
        self._flatten_ys = self._flatten(as_tensors(self._ys))
        self._cache = {}

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def _lazy_axis(self):
        """ "The axis of lazily evaluated."""
        raise NotImplementedError

    def _lazy_indexes(self, indexes):
        idx = indexes[self._lazy_axis]
        return (
            (idx,)
            if isinstance(idx, int)
            else tuple(range(idx.start, idx.stop, idx.step))
        )

    def _flatten(self, xs):
        raise NotImplementedError

    def _shifted_indexes(self, indexes, lazy_axis_size=0):
        idx = indexes[self._lazy_axis]
        shifted_lazy_axis_idx = (
            0 if isinstance(idx, int) else slice(0, lazy_axis_size, 1)
        )
        return (
            indexes[: self._lazy_axis]
            + (shifted_lazy_axis_idx,)
            + indexes[self._lazy_axis + 1 :]
        )

    def __getitem__(self, indexes):
        indexes = _multi_index(indexes, self.shape)

        if isinstance(indexes[self._lazy_axis], int):
            other_indexes = (
                indexes[: self._lazy_axis] + indexes[self._lazy_axis + 1 :]
            )
            return self._cached_evaluate(indexes[self._lazy_axis])[
                other_indexes
            ]
        lazy_indexes = self._lazy_indexes(indexes)
        # Using concat and reshape to replace stack operator temporarily, as
        # it is not a primitive operator.
        shape = list(self.shape)
        shape[self._lazy_axis] = len(lazy_indexes)
        part_jac = paddle.concat(
            [self._cached_evaluate(i) for i in lazy_indexes],
            axis=self._lazy_axis,
        ).reshape(shape)
        return part_jac[self._shifted_indexes(indexes, len(lazy_indexes))]

    def _cached_evaluate(self, k):
        v = self._cache.get(k)
        if v is None:
            v = self._evaluate(k)
            self._cache[k] = v
        return v

    def _evaluate(self, index):
        """Evaluate one slice at along lazy axis."""
        raise NotImplementedError


class _JacobianNoBatch(_Jacobian):
    """Compute Jacobian matrix without batch dimension.
    Suppose the mapping is :math:`f: R^M \to R^N`, the output shape is
    ``(N, M)`` .
    """

    def __init__(self, ys, xs):
        super().__init__(ys, xs)

    @property
    def shape(self):
        return (self._flatten_ys.shape[0], self._flatten_xs.shape[0])

    @property
    def _lazy_axis(self):
        return 0

    def _flatten(self, xs):
        return paddle.concat(tuple(x.reshape((-1,)) for x in xs))

    def _evaluate(self, row_index):
        return self._flatten(
            _grad(
                self._flatten_ys[row_index],
                self._xs,
            )
        )


class _JacobianBatchFirst(_Jacobian):
    """Compute Jacobian matrix with batch at first axis.
    Suppose the mapping is :math:`f: R^{B,M} \to R^{B,N}`, the output shape is
    ``(B, N, M)`` .
    """

    def __init__(self, ys, xs):
        super().__init__(ys, xs)

    @property
    def shape(self):
        return (
            self._flatten_xs.shape[0],
            self._flatten_ys.shape[1],
            self._flatten_xs.shape[1],
        )

    @property
    def _lazy_axis(self):
        return 1

    def _flatten(self, xs):
        return paddle.concat(
            tuple(x.reshape((x.shape[0], -1)) for x in as_tensors(xs)), 1
        )

    def _evaluate(self, row_index):
        return self._flatten(_grad(self._flatten_ys[:, row_index], self._xs))


def _multi_index(indexes, shape):
    """A tool for parsing N-dimensional index into a standard format.

    Currently supporting following input format:
        * ([positive|negative|slice], ...), the right-most elements can be
            omited.

    The standard format after converted is slice tuple which contains N elements:
        * ([positive|slice], ..., [positive|slice])

    Notes:
        Ellipsis indexes such as ``(..., i), (i, ...)`` is not supported.

    Args:
        indexes (tuple): The input indexes.
        shape (tuple): The input shape.

    Returns:
        tuple: The standard format index as the above description.
    """
    indexes = indexes if isinstance(indexes, Sequence) else (indexes,)
    if any(isinstance(i, type(Ellipsis)) for i in indexes):
        raise IndexError('Ellipsis index currently is not supported.')
    # Fill the right-most elements.
    indexes = indexes + (slice(0, None, None),) * (len(shape) - len(indexes))
    # Convert to positive index.
    positive_indexes = []
    for i, index in enumerate(indexes):
        if isinstance(index, slice):
            index = slice(
                index.start or 0, index.stop or shape[i], index.step or 1
            )
            positive_indexes.append(
                slice(
                    index.start + shape[i] if index.start < 0 else index.start,
                    index.stop + shape[i] if index.stop < 0 else index.stop,
                    # Negative step means index backward, no need to convert to
                    # positive interger.
                    index.step,
                )
            )
        elif isinstance(index, int):
            positive_indexes.append(index + shape[i] if index < 0 else index)
        else:
            raise TypeError(f'Not supported index type {index}.')
    return tuple(positive_indexes)


def jacobian(
    ys: Tuple[paddle.Tensor, ...],
    xs: Tuple[paddle.Tensor, ...],
    batch_axis: Optional[int] = None,
) -> Union[List[List[Jacobian]], List[Jacobian], Jacobian]:
    """Function that computes the jacobian of ys deriveted from xs.

    Args:
        ys (Tuple[paddle.Tensor, ...]): Output or list of outputs derived from xs.
        xs (Tuple[paddle.Tensor, ...]): Input or list of inputs.
        batch_axis (Optional[int], optional): Index of batch axis. Defaults to None.

    Returns:
        Union[List[List[Jacobian]], List[Jacobian], Jacobian]: Jacobian(s) of ys
            deriveted from xs.
    """
    if not isinstance(batch_axis, (int, None)):
        raise ValueError(
            f"batch_axis should be None or int, but got {type(batch_axis)}."
        )
    # TODO(HydrogenSulfate): support batch_axis > 0
    if isinstance(batch_axis, int) and batch_axis != 0:
        raise ValueError("Only support batch_axis=0 yet.")

    is_batched = batch_axis is not None

    if isinstance(ys, Sequence) and isinstance(xs, Sequence):
        _jacobian = [
            [Jacobian(_ys, _xs, is_batched) for _xs in xs] for _ys in ys
        ]
    elif isinstance(ys, Sequence) and not isinstance(xs, Sequence):
        _jacobian = [Jacobian(_ys, xs, is_batched) for _ys in ys]
    elif not isinstance(ys, Sequence) and isinstance(xs, Sequence):
        _jacobian = [Jacobian(ys, _xs, is_batched) for _xs in xs]
    else:
        _jacobian = Jacobian(ys, xs, is_batched)

    return _jacobian


def hessian(
    ys: Tuple[paddle.Tensor, ...],
    xs: Tuple[paddle.Tensor, ...],
    batch_axis: Optional[int] = None,
) -> Union[List[List[Hessian]], List[Hessian], Hessian]:
    """Function that computes the hessians of ys deriveted from xs.

    Args:
        ys (Tuple[paddle.Tensor, ...]): Output or list of outputs derived from xs.
        xs (Tuple[paddle.Tensor, ...]): Input or list of inputs.
        batch_axis (Optional[int], optional): Index of batch axis. Defaults to None.

    Returns:
        Union[List[List[Hessian]], List[Hessian], Hessian]: Hessian(s) of ys
            deriveted from xs.
    """
    if not isinstance(batch_axis, (int, None)):
        raise ValueError(
            f"batch_axis should be None or int, but got {type(batch_axis)}."
        )
    # TODO(HydrogenSulfate): support batch_axis > 0
    if isinstance(batch_axis, int) and batch_axis != 0:
        raise ValueError("Only support batch_axis=0 yet.")

    is_batched = batch_axis is not None

    if isinstance(ys, Sequence) and isinstance(xs, Sequence):
        _hessian = [[Hessian(_ys, _xs, is_batched) for _xs in xs] for _ys in ys]
    elif isinstance(ys, Sequence) and not isinstance(xs, Sequence):
        _hessian = [Hessian(_ys, xs, is_batched) for _ys in ys]
    elif not isinstance(ys, Sequence) and isinstance(xs, Sequence):
        _hessian = [Hessian(ys, _xs, is_batched) for _xs in xs]
    else:
        _hessian = Hessian(ys, xs, is_batched)

    return _hessian


def _replace_none_with_zero_tensor(xs, refs):
    if xs is None:
        xs = paddle.zeros_like(refs)
        xs.stop_gradient = refs.stop_gradient
        return xs
    elif isinstance(xs, Sequence):
        return tuple(
            _replace_none_with_zero_tensor(x, refs[i]) for i, x in enumerate(xs)
        )
    else:
        return xs


def _grad(ys, xs, v=None):
    """A gradient function that can be used in dynamic graph and static graph.

    The ``grad`` combines ``paddle.grad`` used in dynamic graph and
    ``paddle.static.gradients`` used in static graph, and do following changes:

    * The ``allow_unused`` flag is removed and set defaults to true internally,
        none in outputs will be replaced by zero tensor.
    * The ``create_graph`` flag is removed and set defaults to true internally,
        only makes sense in dynamic graph.
    * When xs is a single Tensor, ``paddle.grad`` returns a list which only
        contains one Tensor. It may confuse users, thus in this case we improve
        to return a single Tensor in _grad interface.

    Args:
        ys (Tensor|Sequence[Tensor]): The output tensor or tensor sequence of
            the graph to compute gradients.
        xs (Tensor|Sequence[Tensor]): The input tensor or tensor sequence of the graph to
            compute gradients. The returned values of this API are the
            gradients of inputs .
        v (Tensor|Sequence[Tensor]|None,optional): The initial gradient values
            of outputs . If grad_outputs is None, the initial gradient values of
            outputs would be Tensors filled with 1; if grad_outputs is not None,
            it must have the same length as outputs , and in this case, the
            initial gradient value of the i-th outputs would be: (1) a Tensor
            filled with 1 when the i-th element of grad_outputs is None;
            (2) the i-th element of grad_outputs when the i-th element of
            grad_outputs is a Tensor. Default None.

    Returns:
        Tensor|tuple[Tensor]: Tensor or a tuple of Tensors, whose length is the
            same as the Tensor number inside inputs, and the i-th returned
            Tensor is the sum of gradients of outputs with respect to the i-th
            inputs.
    """
    if paddle.fluid._non_static_mode():
        # paddle.grad returns a list though the inputs is a signle Tensor. The
        # follow code snippet fixes the problem by return the first element of
        # xs_grad when the xs is a signle Tensor.
        xs_grad = paddle.grad(ys, xs, v, create_graph=True, allow_unused=True)
        if (
            isinstance(xs, paddle.fluid.framework.Variable)
            and isinstance(xs_grad, Sequence)
            and len(xs_grad) > 0
        ):
            xs_grad = xs_grad[0]
    else:
        xs_grad = paddle.incubate.autograd.grad(ys, xs, v)
    return _replace_none_with_zero_tensor(xs_grad, xs)
