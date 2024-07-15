#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

from typing import TYPE_CHECKING, Generator, Literal, overload

import paddle
import paddle.nn.functional as F
from paddle import _C_ops, in_dynamic_mode
from paddle.base.framework import in_dynamic_or_pir_mode
from paddle.base.layer_helper import LayerHelper
from paddle.base.wrapped_decorator import signature_safe_contextmanager

g_enable_math = None
g_enable_flash = None
g_enable_mem_efficient = None

if TYPE_CHECKING:
    from paddle import Tensor


@signature_safe_contextmanager
def sdp_kernel(
    enable_math: bool = False,
    enable_flash: bool = True,
    enable_mem_efficient: bool = True,
) -> Generator[None, None, None]:
    r"""
    With the sdp_kernel context manager, different algorithm implementations can
    be selected for scaled_dot_product_attention.
    """
    global g_enable_math, g_enable_flash, g_enable_mem_efficient
    original_enable_math = g_enable_math
    original_enable_flash = g_enable_math
    original_enable_mem_efficient = g_enable_mem_efficient

    g_enable_math = enable_math
    g_enable_flash = enable_flash
    g_enable_mem_efficient = enable_mem_efficient
    try:
        yield
    finally:
        g_enable_math = original_enable_math
        g_enable_flash = original_enable_flash
        g_enable_mem_efficient = original_enable_mem_efficient


# special for XPU device
def get_triangle_upper_mask(x: Tensor) -> Tensor:
    mask = paddle.full_like(x, -1e4)
    mask.stop_gradient = True
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask


@overload
def _math_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout_rate: float = ...,
    causal: bool = ...,
    return_softmax: Literal[False] = ...,
    training: bool = ...,
) -> tuple[Tensor, None]:
    ...


@overload
def _math_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout_rate: float = ...,
    causal: bool = ...,
    return_softmax: Literal[True] = ...,
    training: bool = ...,
) -> tuple[Tensor, Tensor]:
    ...


@overload
def _math_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout_rate: float = ...,
    causal: bool = ...,
    return_softmax: bool = ...,
    training: bool = ...,
) -> tuple[Tensor, Tensor | None]:
    ...


def _math_attention(
    query,
    key,
    value,
    dropout_rate=0.0,
    causal=False,
    return_softmax=False,
    training=True,
):
    r"""
    This is a basic implementation of scaled dot product attention composed of
    combinations of fundamental components.
    """
    head_dim = query.shape[-1]
    query = paddle.transpose(query, [0, 2, 1, 3])
    key = paddle.transpose(key, [0, 2, 1, 3])
    value = paddle.transpose(value, [0, 2, 1, 3])
    product = paddle.matmul(
        x=query * (head_dim**-0.5), y=key, transpose_y=True
    )

    if not causal:
        weights = F.softmax(product)
    else:
        # special for XPU device
        place = paddle.get_device()
        if "xpu" in place:
            # softmax_mask_fuse_upper_triangle is not supported on XPU, use plain implementation
            mask = get_triangle_upper_mask(product)
            product = product + mask
            weights = F.softmax(product)
        else:
            weights = paddle.incubate.softmax_mask_fuse_upper_triangle(product)
    if dropout_rate > 0.0:
        weights = F.dropout(
            weights, dropout_rate, training=training, mode="upscale_in_train"
        )

    out = paddle.matmul(weights, value)
    out = paddle.transpose(out, [0, 2, 1, 3])
    return out, weights if return_softmax else None


def _select_sdp_cuda(head_dim: int) -> str:
    if head_dim <= 256:
        return "flash_attn"
    else:
        return "mem_efficient"


def _select_sdp(head_dim: int) -> str:
    r"""
    There are currently three different implementation options available for
    scaled dot product attention, and the chosen approach depends on whether it
    is determined by the sdp_kernel configuration or specified through input values.
    """
    place = paddle.get_device()

    if "xpu" in place:
        return "flash_attn"

    # not use sdp_kernel
    if g_enable_flash is None:
        if "gpu" not in place:
            return "math"
        else:
            return _select_sdp_cuda(head_dim)

    if (
        g_enable_math is False
        and g_enable_flash is False
        and g_enable_mem_efficient is False
    ):
        raise AssertionError(
            "No available backend for scaled_dot_product_attention was found."
        )

    if g_enable_math is True:
        if g_enable_flash is False and g_enable_mem_efficient is False:
            return "math"
        if "gpu" not in place:
            return "math"
    if g_enable_flash is True and g_enable_mem_efficient is True:
        return _select_sdp_cuda(head_dim)
    if g_enable_flash is True:
        return "flash_attn"
    return "mem_efficient"


@overload
def flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: Literal[False] = ...,
    *,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, None]:
    ...


@overload
def flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: Literal[True] = ...,
    *,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor]:
    ...


@overload
def flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: bool = ...,
    *,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor | None]:
    ...


def flash_attention(
    query,
    key,
    value,
    dropout=0.0,
    causal=False,
    return_softmax=False,
    *,
    fixed_seed_offset=None,
    rng_name="",
    training=True,
    name=None,
):
    r"""
    The equation is:

    .. math::

        result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

    where : ``Q``, ``K``, and ``V`` represent the three input parameters of the attention module.
    The dimensions of the three parameters are the same.
    ``d`` represents the size of the last dimension of the three parameters.

    Warning:
        This API is only support inputs with dtype float16 and bfloat16.

    Args:
        query(Tensor): The query tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        key(Tensor): The key tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        value(Tensor): The value tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        dropout(float): The dropout ratio.
        causal(bool): Whether enable causal mode.
        return_softmax(bool): Whether to return softmax.
        fixed_seed_offset(Tensor|None, optional): With fixed seed, offset for dropout mask.
        training(bool): Whether it is in the training phase.
        rng_name(str): The name to select Generator.
        name(str|None, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        out(Tensor): The attention tensor.
                    4-D tensor with shape: [batch_size, seq_len, num_heads, head_dim].
                    The dtype can be float16 or bfloat16.
        softmax(Tensor): The softmax tensor. None if return_softmax is False.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.seed(2023)
            >>> q = paddle.rand((1, 128, 2, 16))

            >>> output = paddle.nn.functional.flash_attention.flash_attention(q, q, q, 0.9, False, False)
            >>> print(output)
            (Tensor(shape=[1, 128, 2, 16], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[0.34992966, 0.34456208, 0.45826620, ..., 0.39883569,
                0.42132431, 0.39157745],
               [0.76687670, 0.65837246, 0.69117945, ..., 0.82817286,
                0.76690865, 0.71485823]],
              ...,
              [[0.71662450, 0.57275224, 0.57053083, ..., 0.48108247,
                0.53336465, 0.54540104],
               [0.59137970, 0.51350880, 0.50449550, ..., 0.38860250,
                0.40526697, 0.60541755]]]]), None)

    """
    head_dim = query.shape[3]
    sdp_func_name = _select_sdp(head_dim)

    if sdp_func_name == "flash_attn":
        if in_dynamic_or_pir_mode():
            (result_attention, result_softmax, _, _) = _C_ops.flash_attn(
                query,
                key,
                value,
                fixed_seed_offset,
                None,
                dropout,
                causal,
                return_softmax,
                not training,
                rng_name,
            )
            return result_attention, result_softmax if return_softmax else None

        helper = LayerHelper('flash_attn', **locals())
        dtype = helper.input_dtype(input_param_name='q')
        out = helper.create_variable_for_type_inference(dtype)
        softmax = helper.create_variable_for_type_inference(dtype)
        softmax_lse = helper.create_variable_for_type_inference(paddle.float32)
        seed_offset = helper.create_variable_for_type_inference(paddle.int64)
        inputs = {
            'q': query,
            'k': key,
            'v': value,
            'fixed_seed_offset': fixed_seed_offset,
        }
        outputs = {
            'out': out,
            'softmax': softmax,
            'softmax_lse': softmax_lse,
            'seed_offset': seed_offset,
        }
        helper.append_op(
            type='flash_attn',
            inputs=inputs,
            outputs=outputs,
            attrs={
                'dropout': dropout,
                'causal': causal,
                'return_softmax': return_softmax,
                'is_test': not training,
                'rng_name': rng_name,
            },
        )
        return out, softmax if return_softmax else None
    else:
        if sdp_func_name == "mem_efficient":
            from paddle.incubate.nn.memory_efficient_attention import (
                memory_efficient_attention,
            )

            output = memory_efficient_attention(
                query,
                key,
                value,
                attn_bias=None,
                p=dropout,
                scale=None,
                training=training,
            )
            return output, None
        else:
            return _math_attention(
                query,
                key,
                value,
                dropout_rate=dropout,
                causal=causal,
                return_softmax=return_softmax,
                training=training,
            )


@overload
def flash_attn_qkvpacked(
    qkv: Tensor,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: Literal[False] = ...,
    *,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, None]:
    ...


@overload
def flash_attn_qkvpacked(
    qkv: Tensor,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: Literal[True] = ...,
    *,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor]:
    ...


@overload
def flash_attn_qkvpacked(
    qkv: Tensor,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: bool = ...,
    *,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor | None]:
    ...


def flash_attn_qkvpacked(
    qkv,
    dropout=0.0,
    causal=False,
    return_softmax=False,
    *,
    fixed_seed_offset=None,
    rng_name="",
    training=True,
    name=None,
):
    r"""
    The equation is:

    .. math::

        result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

    where : ``Q``, ``K``, and ``V`` represent the three input parameters of the attention module.
    The dimensions of the three parameters are the same.
    ``d`` represents the size of the last dimension of the three parameters.

    Warning:
        This API only supports inputs with dtype float16 and bfloat16.
        Don't call this API if flash_attn is not supported.

    Args:
        qkv(Tensor): The query/key/value packed tensor in the Attention module.
                        5-D tensor with shape:
                        [batchsize, seqlen , num_heads/num_heads_k + 2, num_heads_k, head_dim].
                        The dtype can be float16 or bfloat16.
        dropout(float): The dropout ratio.
        causal(bool): Whether enable causal mode.
        return_softmax(bool): Whether to return softmax.
        fixed_seed_offset(Tensor|None, optional): With fixed seed, offset for dropout mask.
        training(bool): Whether it is in the training phase.
        rng_name(str): The name to select Generator.
        name(str|None, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        - out(Tensor). The attention tensor. 4-D tensor with shape: [batch_size, seq_len, num_heads, head_dim]. The dtype can be float16 or bfloat16.
        - softmax(Tensor). The softmax tensor. None if return_softmax is False.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('flash_attn need A100 compile')
            >>> import paddle

            >>> paddle.seed(2023)
            >>> q = paddle.rand((1, 128, 2, 16))
            >>> qkv = paddle.stack([q, q, q], axis=2)
            >>> output = paddle.nn.functional.flash_attn_qkvpacked(qkv, 0.9, False, False)
            >>> print(output)
            (Tensor(shape=[1, 128, 2, 16], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[0.34992966, 0.34456208, 0.45826620, ..., 0.39883569,
                0.42132431, 0.39157745],
               [0.76687670, 0.65837246, 0.69117945, ..., 0.82817286,
                0.76690865, 0.71485823]],
              ...,
              [[0.71662450, 0.57275224, 0.57053083, ..., 0.48108247,
                0.53336465, 0.54540104],
               [0.59137970, 0.51350880, 0.50449550, ..., 0.38860250,
                0.40526697, 0.60541755]]]]), None)
            >>> # doctest: -SKIP

    """
    head_dim = qkv.shape[-1]
    sdp_func_name = _select_sdp(head_dim)

    if sdp_func_name == "flash_attn":
        if in_dynamic_or_pir_mode():
            (
                result_attention,
                result_softmax,
                _,
                _,
            ) = _C_ops.flash_attn_qkvpacked(
                qkv,
                fixed_seed_offset,
                None,
                dropout,
                causal,
                return_softmax,
                not training,
                rng_name,
            )
            return result_attention, result_softmax if return_softmax else None

        helper = LayerHelper('flash_attn_qkvpacked', **locals())
        dtype = helper.input_dtype(input_param_name='qkv')
        out = helper.create_variable_for_type_inference(dtype)
        softmax = helper.create_variable_for_type_inference(dtype)
        softmax_lse = helper.create_variable_for_type_inference(paddle.float32)
        seed_offset = helper.create_variable_for_type_inference(paddle.int64)
        inputs = {
            'qkv': qkv,
            'fixed_seed_offset': fixed_seed_offset,
        }
        outputs = {
            'out': out,
            'softmax': softmax,
            'softmax_lse': softmax_lse,
            'seed_offset': seed_offset,
        }
        helper.append_op(
            type='flash_attn_qkvpacked',
            inputs=inputs,
            outputs=outputs,
            attrs={
                'dropout': dropout,
                'causal': causal,
                'return_softmax': return_softmax,
                'is_test': not training,
                'rng_name': rng_name,
            },
        )
        return out, softmax if return_softmax else None
    else:
        # don't call qkvpacked if not using flash_attn
        query = qkv[:, :, :-2].reshape([0, 0, -1, qkv.shape[-1]])
        key = qkv[:, :, -2]
        value = qkv[:, :, -1]
        if sdp_func_name == "mem_efficient":
            from paddle.incubate.nn.memory_efficient_attention import (
                memory_efficient_attention,
            )

            output = memory_efficient_attention(
                query,
                key,
                value,
                attn_bias=None,
                p=dropout,
                scale=None,
                training=training,
            )
            return output, None
        else:
            return _math_attention(
                query,
                key,
                value,
                dropout_rate=dropout,
                causal=causal,
                return_softmax=return_softmax,
                training=training,
            )


@overload
def flash_attn_unpadded(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: Literal[False] = ...,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, None]:
    ...


@overload
def flash_attn_unpadded(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: Literal[True] = ...,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor]:
    ...


@overload
def flash_attn_unpadded(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: bool = ...,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor | None]:
    ...


def flash_attn_unpadded(
    query,
    key,
    value,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    scale,
    dropout=0.0,
    causal=False,
    return_softmax=False,
    fixed_seed_offset=None,
    rng_name='',
    training=True,
    name=None,
):
    r"""
    The equation is:

    .. math::

        result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

    where : ``Q``, ``K``, and ``V`` represent the three input parameters of the attention module.
    The dimensions of the three parameters are the same.
    ``d`` represents the size of the last dimension of the three parameters.

    Warning:
        This API is only support inputs with dtype float16 and bfloat16.

    Args:
        query(Tensor): The query tensor in the Attention module.
                        3-D tensor with shape:
                        [total_seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        key(Tensor): The key tensor in the Attention module.
                        3-D tensor with shape:
                        [total_seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        value(Tensor): The value tensor in the Attention module.
                        3-D tensor with shape:
                        [total_seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        cu_seqlens_q(Tensor): The cumulative sequence lengths of the sequences in the batch,
                        used to index query.
        cu_seqlens_k(Tensor): The cumulative sequence lengths of the sequences in the batch,
                        used to index key and value.
        max_seqlen_q(int): Maximum sequence length of query in the batch.
        max_seqlen_k(int): Maximum sequence length of key/value in the batch.
        scale(float): The scaling of QK^T before applying softmax.
        dropout(float, optional): The dropout ratio.
        causal(bool, optional): Whether enable causal mode.
        return_softmax(bool, optional): Whether to return softmax.
        fixed_seed_offset(Tensor|None, optional): With fixed seed, offset for dropout mask.
        rng_name(str, optional): The name to select Generator.
        training(bool, optional): Whether it is in the training phase.
        name(str|None, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        out(Tensor): The attention tensor.
                    3-D tensor with shape: [total_seq_len, num_heads, head_dim].
                    The dtype can be float16 or bfloat16.
        softmax(Tensor): The softmax tensor. None if return_softmax is False.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)
            >>> q = paddle.rand((2, 128, 8, 16), dtype='float16')
            >>> cu = paddle.arange(0, 384, 128, dtype='int32')
            >>> qq = paddle.reshape(q, [256, 8, 16])
            >>> output = paddle.nn.functional.flash_attention.flash_attn_unpadded(qq, qq, qq, cu, cu, 128, 128, 0.25, 0.0, False, False)

    """
    if in_dynamic_mode():
        (
            result_attention,
            result_softmax,
        ) = _C_ops.flash_attn_unpadded(
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_k,
            fixed_seed_offset,
            None,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            dropout,
            causal,
            return_softmax,
            not training,
            rng_name,
        )
        return result_attention, result_softmax if return_softmax else None

    helper = LayerHelper('flash_attn_unpadded', **locals())
    dtype = helper.input_dtype(input_param_name='q')
    out = helper.create_variable_for_type_inference(dtype)
    softmax = helper.create_variable_for_type_inference(dtype)
    softmax_lse = helper.create_variable_for_type_inference(paddle.float32)
    seed_offset = helper.create_variable_for_type_inference(paddle.int64)
    inputs = {
        'q': query,
        'k': key,
        'v': value,
        'cu_seqlens_q': cu_seqlens_q,
        'cu_seqlens_k': cu_seqlens_k,
        'fixed_seed_offset': fixed_seed_offset,
    }
    outputs = {
        'out': out,
        'softmax': softmax,
        'softmax_lse': softmax_lse,
        'seed_offset': seed_offset,
    }
    helper.append_op(
        type='flash_attn_unpadded',
        inputs=inputs,
        outputs=outputs,
        attrs={
            'max_seqlen_q': max_seqlen_q,
            'max_seqlen_k': max_seqlen_k,
            'scale': scale,
            'dropout': dropout,
            'causal': causal,
            'return_softmax': return_softmax,
            'is_test': not training,
            'rng_name': rng_name,
        },
    )
    return out, softmax if return_softmax else None


@overload
def flash_attn_varlen_qkvpacked(
    qkv: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: Literal[False] = ...,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    varlen_padded: bool = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, None]:
    ...


@overload
def flash_attn_varlen_qkvpacked(
    qkv: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: Literal[True] = ...,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    varlen_padded: bool = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor]:
    ...


@overload
def flash_attn_varlen_qkvpacked(
    qkv: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: bool = ...,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    varlen_padded: bool = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor | None]:
    ...


def flash_attn_varlen_qkvpacked(
    qkv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    scale,
    dropout=0.0,
    causal=False,
    return_softmax=False,
    fixed_seed_offset=None,
    rng_name="",
    varlen_padded=True,
    training=True,
    name=None,
):
    r"""
    The equation is:

    .. math::

        result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

    where : ``Q``, ``K``, and ``V`` represent the three input parameters of the attention module.
    The dimensions of the three parameters are the same.
    ``d`` represents the size of the last dimension of the three parameters.

    Warning:
        This API only supports inputs with dtype float16 and bfloat16.

    Args:
        qkv(Tensor): The padded query/key/value packed tensor in the Attention module. The padding part won't be computed
                        4-D tensor with shape:
                        [total_seq_len, num_heads/num_heads_k + 2, num_heads_k, head_dim].
                        The dtype can be float16 or bfloat16.
        cu_seqlens_q(Tensor): The cumulative sequence lengths of the sequences in the batch,
                        used to index query.
        cu_seqlens_k(Tensor): The cumulative sequence lengths of the sequences in the batch,
                        used to index key and value.
        max_seqlen_q(int): Maximum sequence length of query in the batch. Note it's the padding length, not the max actual seqlen
        max_seqlen_k(int): Maximum sequence length of key/value in the batch.
        scale(float): The scaling of QK^T before applying softmax.
        dropout(float, optional): The dropout ratio.
        causal(bool, optional): Whether enable causal mode.
        return_softmax(bool, optional): Whether to return softmax.
        fixed_seed_offset(Tensor|None, optional): With fixed seed, offset for dropout mask.
        rng_name(str, optional): The name to select Generator.
        training(bool, optional): Whether it is in the training phase.
        name(str|None, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        - out(Tensor). The attention tensor. The tensor is padded by zeros. 3-D tensor with shape: [total_seq_len, num_heads, head_dim]. The dtype can be float16 or bfloat16.
        - softmax(Tensor). The softmax tensor. None if return_softmax is False.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('flash_attn need A100 compile')
            >>> import paddle
            >>> paddle.seed(2023)
            >>> q = paddle.rand((2, 128, 8, 16), dtype='float16')
            >>> cu = paddle.arange(0, 384, 128, dtype='int32')
            >>> qq = paddle.reshape(q, [256, 8, 16])
            >>> qkv = paddle.stack([qq, qq, qq], axis=2)
            >>> output = paddle.nn.functional.flash_attn_varlen_qkvpacked(qkv, cu, cu, 128, 128, 0.25, 0.0, False, False)
            >>> # doctest: -SKIP

    """
    if in_dynamic_mode():
        (
            result_attention,
            result_softmax,
        ) = _C_ops.flash_attn_varlen_qkvpacked(
            qkv,
            cu_seqlens_q,
            cu_seqlens_k,
            fixed_seed_offset,
            None,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            dropout,
            causal,
            return_softmax,
            not training,
            rng_name,
            varlen_padded,
        )
        return result_attention, result_softmax if return_softmax else None

    helper = LayerHelper('flash_attn_varlen_qkvpacked', **locals())
    dtype = helper.input_dtype(input_param_name='qkv')
    out = helper.create_variable_for_type_inference(dtype)
    softmax = helper.create_variable_for_type_inference(dtype)
    softmax_lse = helper.create_variable_for_type_inference(paddle.float32)
    seed_offset = helper.create_variable_for_type_inference(paddle.int64)
    inputs = {
        'qkv': qkv,
        'cu_seqlens_q': cu_seqlens_q,
        'cu_seqlens_k': cu_seqlens_k,
        'fixed_seed_offset': fixed_seed_offset,
    }
    outputs = {
        'out': out,
        'softmax': softmax,
        'softmax_lse': softmax_lse,
        'seed_offset': seed_offset,
    }
    helper.append_op(
        type='flash_attn_varlen_qkvpacked',
        inputs=inputs,
        outputs=outputs,
        attrs={
            'max_seqlen_q': max_seqlen_q,
            'max_seqlen_k': max_seqlen_k,
            'scale': scale,
            'dropout': dropout,
            'causal': causal,
            'return_softmax': return_softmax,
            'is_test': not training,
            'rng_name': rng_name,
        },
    )
    return out, softmax if return_softmax else None


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    training: bool = True,
    scaling_factor: float | None = None,
    name: str | None = None,
) -> Tensor:
    r"""
    The equation is:

    .. math::

        result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

    where : ``Q``, ``K``, and ``V`` represent the three input parameters of the attention module.
    The dimensions of the three parameters are the same.
    ``d`` represents the size of the last dimension of the three parameters.

    Warning:
        This API only supports inputs with dtype float16 and bfloat16.

    Args:
        query(Tensor): The query tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        key(Tensor): The key tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        value(Tensor): The value tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        attn_mask(Tensor|None, optional): A float mask of the same type as query,
                        key, value that is added to the attention score.
        dropout_p(float, optional): The dropout ratio.
        is_causal(bool, optional): Whether enable causal mode.
        training(bool, optional): Whether it is in the training phase.
        scaling_factor (float|None, optional): The scaling factor for the attention scores.
                        Currently only fused_dot_product_attention supports
                        custom scaling_factor.
        name(str|None, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        out(Tensor): The attention tensor.
                    4-D tensor with shape: [batch_size, seq_len, num_heads, head_dim].
                    The dtype can be float16 or bfloat16.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('bfloat need V100 compile')
            >>> import paddle
            >>> q = paddle.rand((1, 128, 2, 16), dtype=paddle.bfloat16)
            >>> output = paddle.nn.functional.scaled_dot_product_attention(q, q, q, None, 0.9, False)
            >>> print(output)
            >>> # doctest: -SKIP
    """

    def is_device_hopper() -> bool:
        if paddle.device.is_compiled_with_cuda():
            prop = paddle.device.cuda.get_device_properties()
            sm_version = prop.major * 10 + prop.minor
            return sm_version >= 90
        return False

    # NOTE: fused_dot_product_attention runs faster on Hopper devices
    # but slower on Ampere devices.
    if is_device_hopper():
        from paddle.incubate.nn.functional import fused_dot_product_attention

        return fused_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            training=training,
            scaling_factor=scaling_factor,
            name=name,
        )

    if attn_mask is None:
        # downgraded to ordinary flash attention implementation
        out, _ = flash_attention(query, key, value, dropout_p, is_causal)
        return out
    else:
        if in_dynamic_or_pir_mode():
            fixed_seed_offset = None
            return_softmax = False
            rng_name = ""
            out, _, _, _ = _C_ops.flash_attn(
                query,
                key,
                value,
                fixed_seed_offset,
                attn_mask,
                dropout_p,
                is_causal,
                return_softmax,
                not training,
                rng_name,
            )
            return out
        else:
            helper = LayerHelper('flash_attn', **locals())
            dtype = helper.input_dtype(input_param_name='q')
            out = helper.create_variable_for_type_inference(dtype)
            softmax = helper.create_variable_for_type_inference(dtype)
            softmax_lse = helper.create_variable_for_type_inference(
                paddle.float32
            )
            seed_offset = helper.create_variable_for_type_inference(
                paddle.int64
            )
            inputs = {
                'q': query,
                'k': key,
                'v': value,
                'attn_mask': attn_mask,
            }
            outputs = {
                'out': out,
                'softmax': softmax,
                'softmax_lse': softmax_lse,
                'seed_offset': seed_offset,
            }
            helper.append_op(
                type='flash_attn',
                inputs=inputs,
                outputs=outputs,
                attrs={
                    'dropout': dropout_p,
                    'causal': is_causal,
                    'return_softmax': False,
                    'is_test': not training,
                    'rng_name': '',
                },
            )
            return out


def flash_attention_with_sparse_mask(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask_start_row_indices: Tensor,
    attn_mask_start_row: int = 0,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_softmax: bool = False,
    return_softmax_lse: bool = False,
    return_seed_offset: bool = False,
    training: bool = True,
    name: str | None = None,
) -> Tensor:
    r"""
    The equation is:

    .. math::
        result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

    where : ``Q``, ``K``, and ``V`` represent the three input parameters of the attention module.
    The dimensions of the three parameters are the same.
    ``d`` represents the size of the last dimension of the three parameters.

    Warning:
        This API only supports inputs with dtype float16 and bfloat16.

    Args:
        query(Tensor): The query tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        key(Tensor): The key tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        value(Tensor): The value tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        attn_mask_start_row_indices(Tensor): A sparse attention mask
                        indices tensor, the shape is [batch_size, num_head, seq_len],
                        The value of each element indicates the row index where the
                        mask starts in score matrix. The dtype must be int32.
        attn_mask_start_row(int, optional): When `attn_mask_start_row_indices` is passed
                        in and the minimum row number is known to be greater than 0,
                        it can set `attn_mask_start_row` for performance improvement.
                        The default value is 0.
        dropout_p(float, optional): The dropout ratio.
        is_causal(bool, optional): Whether enable causal mode.
        training(bool, optional): Whether it is in the training phase.
        name(str|None, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.
    Returns:
        out(Tensor), The attention tensor.
                    4-D tensor with shape: [batch_size, seq_len, num_heads, head_dim].
                    The dtype can be float16 or bfloat16.
    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('bfloat need V100 compile')
            >>> import paddle
            >>> import numpy as np
            >>> def generate_start_rows(bz, num_head, rows, cols, start_row):
            >>>     assert rows == cols, f"rows {rows} must be equal to cols {cols}."
            >>>     start_rows_list = []
            >>>     for bz_idx in range(bz):
            >>>         for head_idx in range(num_head):
            >>>             start_rows = np.array([rows+1] * cols)
            >>>             mask_pos = np.random.choice(cols-1, cols - start_row, replace=False)
            >>>             index = np.arange(start_row, rows)
            >>>             mask_pos = np.concatenate([mask_pos[mask_pos < index - 1], mask_pos[mask_pos >= index - 1]])
            >>>             start_rows[mask_pos] = index
            >>>             start_rows_list.append(start_rows)
            >>>     start_rows_arr = np.array(start_rows_list).reshape([bz, num_head, rows])
            >>>     return start_rows_arr
            >>> q = paddle.rand((1, 128, 2, 16), dtype=paddle.bfloat16)
            >>> attn_mask_start_row = 48
            >>> start_row_indices = generate_start_rows(1, 2, 128, 128, attn_mask_start_row)
            >>> attn_mask_start_row_indices = paddle.to_tensor(start_row_indices, dtype=paddle.int32)
            >>> out = paddle.nn.functional.flash_attention.flash_attention_with_sparse_mask(
            >>>     q, q, q,
            >>>     attn_mask_start_row_indices=attn_mask_start_row_indices,
            >>>     attn_mask_start_row=attn_mask_start_row,
            >>>     dropout_p=0.9,
            >>>     is_causal=True,
            >>> )
            >>> print(out)
            >>> # doctest: -SKIP
    """

    assert (
        attn_mask_start_row_indices is not None
    ), f"attn_mask_start_row_indices must be not None, but got {attn_mask_start_row_indices}"
    assert (
        is_causal is True
    ), f"is_causal must be True when attn_mask_start_row_indices is not None, but got {is_causal}"
    assert (
        attn_mask_start_row_indices.dtype == paddle.int32
    ), f"attn_mask_start_row_indices.dtype must be paddle.int32, but got {attn_mask_start_row_indices.dtype}"
    assert isinstance(
        attn_mask_start_row, int
    ), f"attn_mask_start_row must be int, but got {type(attn_mask_start_row)}"
    assert (
        attn_mask_start_row >= 0
    ), f"Should set attn_mask_start_row >=0 when attn_mask_start_row_indices is not None, but got {attn_mask_start_row}"

    fixed_seed_offset = None
    return_softmax = False
    rng_name = ""

    (
        out,
        result_softmax,
        result_softmax_lse,
        result_seed_offset,
    ) = _C_ops.flash_attn_with_sparse_mask(
        query,
        key,
        value,
        attn_mask_start_row_indices,
        fixed_seed_offset,
        dropout_p,
        is_causal,
        attn_mask_start_row,
        return_softmax,
        not training,
        rng_name,
    )
    outputs = [out]
    if return_softmax:
        outputs += [result_softmax]
    if return_softmax_lse:
        outputs += [result_softmax_lse]
    if return_seed_offset:
        outputs += [result_seed_offset]
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


def calc_reduced_attention_scores(
    query: paddle.Tensor, key: paddle.Tensor, softmax_lse: paddle.Tensor
) -> paddle.Tensor:
    r"""
    The equation is:

    .. math::

        result=reduce\_sum(softmax(\frac{ Q * K^T }{\sqrt{d}}), dim=-2)

    Warning:
        This API only supports inputs with dtype float16 and bfloat16.

    Args:
        query(Tensor): The query tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seqlen_q, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        key(Tensor): The key tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seqlen_k, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        softmax_lse(Tensor): The logsumexp of each row returned by _C_ops.flash_attn().
                        3-D tensor with shape:
                        [batch_size, num_heads, seqlen_q_rounded], where seqlen_q_rounded = ceil(seqlen_q/128).
                        The dtype is float32.
    Returns:
        reduced_attention_scores(Tensor), The reduce sum of attention scores across seqlen_q.
        4-D tensor with shape: [batch_size, num_heads, 1, seqlen_k]. The dtype is float32.
    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('reduce_attn_scores need A100 compile')
            >>> import paddle
            >>> import numpy as np
            >>> import paddle._C_ops as _C_ops
            >>> from paddle.nn.functional.flash_attention import (
            >>>     calc_reduced_attention_scores
            >>> )
            >>> np.random.seed(2024)
            >>> q_shape = (5,1024,16,128)
            >>> k_shape = (5,2048,16,128)
            >>> dtype = 'float16'
            >>> query = np.random.random(q_shape)
            >>> key = np.random.random(k_shape)
            >>> q = paddle.to_tensor(
            >>>     query, place=place, dtype=dtype, stop_gradient=True
            >>> )
            >>> k = paddle.to_tensor(
            >>>     key, place=place, dtype=dtype, stop_gradient=True
            >>> )
            >>> _, _, softmax_lse, _ = _C_ops.flash_attn(
            >>>     q,
            >>>     k,
            >>>     k,
            >>>     (None,), #fixed_seed_offset
            >>>     None, #attn_mask
            >>>     0.0, #dropout
            >>>     False, #causal
            >>>     False, #return_softmax
            >>>     False, #is_test
            >>>     "" #rng_name
            >>> )
            >>> reduced_attn_scores = calc_reduced_attention_scores(
            >>>     q,
            >>>     k,
            >>>     softmax_lse,
            >>> )
            >>> # doctest: -SKIP
    """
    assert (
        query.stop_gradient and key.stop_gradient
    ), 'calc_reduced_attention_scores() is for inference only.'

    if in_dynamic_or_pir_mode():
        reduced_scores = _C_ops.calc_reduced_attn_scores(
            query, key, softmax_lse
        )
        return reduced_scores

    helper = LayerHelper('calc_reduced_attn_scores', **locals())
    reduced_scores = helper.create_variable_for_type_inference(paddle.float32)
    softmax = helper.create_variable_for_type_inference(paddle.float32)
    inputs = {
        'q': query,
        'k': key,
        'softmax_lse': softmax_lse,
    }
    outputs = {
        'reduced_scores': reduced_scores,
    }
    helper.append_op(
        type='calc_reduced_attn_scores',
        inputs=inputs,
        outputs=outputs,
    )
    return reduced_scores
