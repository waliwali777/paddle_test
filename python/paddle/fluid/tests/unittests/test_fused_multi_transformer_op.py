# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
import paddle.nn as nn
import paddle.fluid.core as core
import paddle.nn.functional as F
import paddle.incubate.nn.functional as incubate_f
from paddle.nn.layer.norm import LayerNorm
from paddle.nn.layer.common import Linear, Dropout
from paddle.nn.layer.transformer import _convert_attention_mask
from paddle import tensor
from paddle.fluid import layers
import unittest
from op_test import OpTest
from paddle.fluid.framework import default_main_program
from paddle.fluid.dygraph.layers import Layer
from paddle.fluid.layer_helper import LayerHelper
from paddle.nn.initializer import Constant
from paddle.fluid.framework import _non_static_mode, default_main_program
from paddle import _C_ops

default_main_program().random_seed = 42


def _set_var_distributed(var):
    if var is None:
        return

    var.is_distributed = True

    # NOTE: use current_block and find_var_recursive to support while_loop
    startup_block = paddle.static.default_startup_program().current_block()
    main_block = paddle.static.default_main_program().current_block()
    startup_block._find_var_recursive(var.name).is_distributed = True
    main_block._find_var_recursive(var.name).is_distributed = True


def fused_multi_transformer(x,
                            qkv_weights,
                            qkv_biases,
                            linear_weights,
                            linear_biases,
                            ln_scales,
                            ln_biases,
                            ffn1_weights,
                            ffn1_biases,
                            ffn2_weights,
                            ffn2_biases,
                            ffn_ln_scales,
                            ffn_ln_biases,
                            pre_layer_norm=True,
                            epsilon=1e-05,
                            cache_kvs=None,
                            attn_mask=None,
                            dropout_rate=0.5,
                            training=False,
                            mode='upscale_in_train',
                            ring_id=-1,
                            name=None):
    seed = None
    if mode not in ('downscale_in_infer', 'upscale_in_train'):
        raise ValueError(
            "mode argument should be 'downscale_in_infer' or 'upscale_in_train'")
    mode = 'downgrade_in_infer' if mode == 'downscale_in_infer' else mode  #semantic transfer

    if _non_static_mode():
        if default_main_program().random_seed != 0:
            seed = default_main_program().random_seed
        # pre_ln_mean, pre_ln_variance, pre_ln_out, qkv_out, qkv_bias_out, transpose_out, qk_out,
        # qktv_out, softmax_out, attn_dropout_mask_out, attn_dropout_out, attn_mask_out, fmha_out,
        # linear_out, dropout_mask_out, ln_mean_out, ln_var_out, bias_dropout_residual_out, final_out

        cache_kv_out, final_out = _C_ops.fused_multi_transformer(
            x, ln_scales, ln_biases, qkv_weights, qkv_biases, cache_kvs,
            attn_mask, linear_weights, linear_biases, ffn_ln_scales,
            ffn_ln_biases, ffn1_weights, ffn1_biases, ffn2_weights, ffn2_biases,
            0 if cache_kvs is None else len(cache_kvs), 'pre_layer_norm',
            pre_layer_norm, 'epsilon', epsilon, 'attn_dropout_rate',
            dropout_rate, 'attn_dropout_is_test', not training,
            'attn_dropout_implementation', mode, 'dropout_implementation', mode,
            'ring_id', ring_id)
        if cache_kvs is not None:
            return final_out, cache_kv_out
        return final_out
    else:
        helper = LayerHelper('fused_multi_head_attention', **locals())
        dtype = x.dtype
        # check dtypes
        check_variable_and_dtype(x, 'x', ['float16', 'float32'],
                                 'fused_multi_transformer')
        check_dtype(dtype, 'dtype', ['float16', 'float32'],
                    'fused_multi_transformer')

        # set inputs
        inputs = dict()
        inputs['X'] = [x]
        inputs['LnScale'] = ln_scales
        inputs['LnBias'] = ln_biases
        inputs['QKVW'] = qkv_weights
        if qkv_biases is not None:
            inputs['QKVBias'] = qkv_biases
        inputs['SrcMask'] = attn_mask
        inputs['OutLinearW'] = linear_weights
        if linear_biases is not None:
            inputs['OutLinearBias'] = linear_biases
        if cache_kvs:
            assert len(cache_kvs) == len(qkv_weights)
            inputs['CacheKV'] = cache_kvs

        if (seed is None or seed == 0) and helper.main_program.random_seed != 0:
            seed = helper.main_program.random_seed

        # set attrs
        attrs = {
            'pre_layer_norm': pre_layer_norm,
            'epsilon': epsilon,
            'attn_dropout_rate': dropout_rate,
            'attn_dropout_is_test': not training,
            'attn_dropout_implementation': mode,
            'ring_id': ring_id
        }

        final_out = helper.create_variable_for_type_inference(dtype=dtype)
        cache_kv_out = helper.create_variable_for_type_inference(dtype=dtype)

        helper.append_op(
            type='fused_multi_transformer',
            inputs=inputs,
            outputs={'Out': final_out,
                     'CacheKVOut': cache_kv_out},
            attrs=attrs)

        return (final_out, cache_kv_out) if cache_kvs else final_out


class ParallelFusedMultiTransformer(Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dim_feedforward,
                 dropout_rate=0.5,
                 normalize_before=True,
                 qkv_weight_attrs=None,
                 qkv_bias_attrs=None,
                 linear_weight_attrs=None,
                 linear_bias_attrs=None,
                 ln_scale_attrs=None,
                 ln_bias_attrs=None,
                 ffn1_weight_attrs=None,
                 ffn1_bias_attrs=None,
                 ffn2_weight_attrs=None,
                 ffn2_bias_attrs=None,
                 ffn_ln_scale_attrs=None,
                 ffn_ln_bias_attrs=None,
                 epsilon=1e-5,
                 num_layers=-1,
                 nranks=1,
                 ring_id=-1,
                 name=None):
        super(ParallelFusedMultiTransformer, self).__init__()

        assert embed_dim > 0, ("Expected embed_dim to be greater than 0, "
                               "but recieved {}".format(embed_dim))
        assert num_heads > 0, ("Expected nhead to be greater than 0, "
                               "but recieved {}".format(num_heads))
        assert dim_feedforward > 0, (
            "Expected dim_feedforward to be greater than 0, but recieved {}".
            format(dim_feedforward))

        self.normalize_before = normalize_before
        self._dtype = self._helper.get_default_dtype()
        self._epsilon = epsilon
        self._ring_id = ring_id

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # tensor model parallel
        if nranks > 1:
            assert ring_id != -1
        assert num_heads % nranks == 0
        assert dim_feedforward % nranks == 0
        num_heads = num_heads // nranks
        dim_feedforward = dim_feedforward // nranks
        self._dim_feedforward = dim_feedforward

        if qkv_weight_attrs is not None:
            num_layers = len(qkv_weight_attrs)
        assert num_layers > 0

        self.qkv_weights, self.qkv_biases = [], []
        self.linear_weights, self.linear_biases = [], []
        self.ln_scales, self.ln_biases = [], []
        self.ffn1_weights, self.ffn1_biases = [], []
        self.ffn2_weights, self.ffn2_biases = [], []
        self.ffn_ln_scales, self.ffn_ln_biases = [], []

        def get_attr(attrs, idx):
            if isinstance(attrs, (list, tuple)):
                assert len(attrs) == num_layers
                return attrs[idx]
            return attrs

        for i in range(num_layers):
            qkv_weight_attr = get_attr(qkv_weight_attrs, i)
            qkv_bias_attr = get_attr(qkv_bias_attrs, i)
            linear_weight_attr = get_attr(linear_weight_attrs, i)
            linear_bias_attr = get_attr(linear_bias_attrs, i)
            ln_scale_attr = get_attr(ln_scale_attrs, i)
            ln_bias_attr = get_attr(ln_bias_attrs, i)

            ffn1_weight_attr = get_attr(ffn1_weight_attrs, i)
            ffn1_bias_attr = get_attr(ffn1_bias_attrs, i)
            ffn2_weight_attr = get_attr(ffn2_weight_attrs, i)
            ffn2_bias_attr = get_attr(ffn2_bias_attrs, i)
            ffn_ln_scale_attr = get_attr(ffn_ln_scale_attrs, i)
            ffn_ln_bias_attr = get_attr(ffn_ln_bias_attrs, i)

            qkv_weight = self.create_parameter(
                shape=[3, num_heads, self.head_dim, embed_dim],
                attr=qkv_weight_attr,
                dtype=self._dtype,
                is_bias=False)
            qkv_bias = self.create_parameter(
                shape=[3, num_heads, self.head_dim],
                attr=qkv_bias_attr,
                dtype=self._dtype,
                is_bias=True)
            linear_weight = self.create_parameter(
                shape=[num_heads * self.head_dim, embed_dim],
                attr=linear_weight_attr,
                dtype=self._dtype,
                is_bias=False)
            linear_bias = self.create_parameter(
                shape=[embed_dim],
                attr=linear_bias_attr,
                dtype=self._dtype,
                is_bias=True)
            ln_scale = self.create_parameter(
                attr=ln_scale_attr,
                shape=[embed_dim],
                default_initializer=Constant(value=1.0))
            ln_bias = self.create_parameter(
                attr=ln_bias_attr, shape=[embed_dim], is_bias=True)

            ffn1_weight = self.create_parameter(
                shape=[embed_dim, dim_feedforward],
                attr=ffn1_weight_attr,
                dtype=self._dtype,
                is_bias=False)
            ffn1_bias = self.create_parameter(
                shape=[dim_feedforward],
                attr=ffn1_bias_attr,
                dtype=self._dtype,
                is_bias=True)
            ffn2_weight = self.create_parameter(
                shape=[dim_feedforward, embed_dim],
                attr=ffn2_weight_attr,
                dtype=self._dtype,
                is_bias=False)
            ffn2_bias = self.create_parameter(
                shape=[embed_dim],
                attr=ffn2_bias_attr,
                dtype=self._dtype,
                is_bias=True)
            ffn_ln_scale = self.create_parameter(
                shape=[embed_dim],
                attr=ffn_ln_scale_attr,
                is_bias=False,
                default_initializer=Constant(1.0))
            ffn_ln_bias = self.create_parameter(
                shape=[embed_dim], attr=ffn_ln_bias_attr, is_bias=True)

            # tensor model parallel
            if nranks > 1:
                # column parallel
                _set_var_distributed(qkv_weight)
                _set_var_distributed(qkv_bias)
                _set_var_distributed(ffn1_weight)
                _set_var_distributed(ffn1_bias)
                # row parallel
                _set_var_distributed(linear_weight)
                _set_var_distributed(ffn2_weight)

            self.qkv_weights.append(qkv_weight)
            self.qkv_biases.append(qkv_bias)
            self.linear_weights.append(linear_weight)
            self.linear_biases.append(linear_bias)
            self.ln_scales.append(ln_scale)
            self.ln_biases.append(ln_bias)

            self.ffn1_weights.append(ffn1_weight)
            self.ffn1_biases.append(ffn1_bias)
            self.ffn2_weights.append(ffn2_weight)
            self.ffn2_biases.append(ffn2_bias)
            self.ffn_ln_scales.append(ffn_ln_scale)
            self.ffn_ln_biases.append(ffn_ln_bias)

        self.dropout_rate = dropout_rate
        self.name = name

    def forward(self, query, attn_mask=None, caches=None):
        if caches is not None:
            assert len(caches) == len(self.qkv_weights)
        out = fused_multi_transformer(
            query,
            self.qkv_weights,
            self.qkv_biases,
            self.linear_weights,
            self.linear_biases,
            self.ln_scales,
            self.ln_biases,
            self.ffn1_weights,
            self.ffn1_biases,
            self.ffn2_weights,
            self.ffn2_biases,
            self.ffn_ln_scales,
            self.ffn_ln_biases,
            pre_layer_norm=self.normalize_before,
            epsilon=self._epsilon,
            cache_kvs=caches,
            attn_mask=attn_mask,
            dropout_rate=self.dropout_rate,
            training=self.training,
            mode='upscale_in_train',
            ring_id=self._ring_id,
            name=self.name)
        return out


class TestFusedMultiTransformerOp(OpTest):
    def setUp(self):
        self.config()
        self.generate_input_data()
        paddle.set_default_dtype(self.x_type)
        self.__class__.op_type = "fused_multi_transformer"
        # use autograd to check grad in this unittest.
        self.__class__.no_need_check_grad = False
        self.q_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            self.weight_attr,
            bias_attr=self.bias_attr)
        self.k_proj = Linear(
            self.kdim,
            self.embed_dim,
            self.weight_attr,
            bias_attr=self.bias_attr)
        self.v_proj = Linear(
            self.vdim,
            self.embed_dim,
            self.weight_attr,
            bias_attr=self.bias_attr)
        self.out_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            self.weight_attr,
            bias_attr=self.bias_attr)
        self.ffn1_proj = Linear(
            self.embed_dim,
            4 * self.embed_dim,
            self.weight_attr,
            bias_attr=self.bias_attr)
        self.ffn2_proj = Linear(
            4 * self.embed_dim,
            self.embed_dim,
            self.weight_attr,
            bias_attr=self.bias_attr)
        paddle.set_default_dtype(np.float32)
        self.norm = LayerNorm(self.embed_dim)
        self.ffn_norm = LayerNorm(self.embed_dim)
        paddle.set_default_dtype(self.x_type)
        self.dropout = Dropout(self.dropout_prob, mode="upscale_in_train")
        self.activation = getattr(F, self.act_method)

    def config(self):
        self.x_type = np.float32
        self.attn_mask_type = np.float64
        self.pre_layer_norm = True
        self.has_attn_mask = True
        self.has_cache_kv = False
        self.training = False

        self.layers = 4
        self.batch_size = 8
        self.query_length = 128
        self.cache_length = 128
        self.head_dim = 64
        self.num_heads = 16
        self.embed_dim = self.head_dim * self.num_heads

        self.dropout_prob = 0.0
        self.attn_dropout_prob = 0.0
        self.act_method = 'gelu'
        self.weight_attr = None
        self.bias_attr = None
        self.kdim, self.vdim = self.embed_dim, self.embed_dim
        self.key_length, self.value_length = self.query_length, self.query_length

    def generate_input_data(self):
        self.query = np.random.rand(self.batch_size, self.query_length,
                                    self.embed_dim).astype(self.x_type)
        out_seq_len = self.key_length
        if self.has_cache_kv:
            assert self.training is False, ValueError(
                'cache_kv can only used in inference')
            self.cache_kv = np.random.rand(2, self.batch_size, self.num_heads,
                                           self.cache_length,
                                           self.head_dim).astype(self.x_type)
            out_seq_len += self.cache_length
        else:
            self.cache_kv = None

        if self.has_attn_mask:
            # [B, n_head, seq_len, out_seq_len]
            self.attn_mask = np.ones(
                (self.batch_size, self.num_heads, self.query_length,
                 out_seq_len),
                dtype=self.attn_mask_type)
            if self.attn_mask_type == np.int64:
                self.attn_mask = np.tril(self.attn_mask)
            elif self.attn_mask_type == np.float64:
                self.attn_mask = (np.tril(self.attn_mask) - 1.0) * 1e9
            else:
                raise ValueError(
                    "'attn_mask_type' should be 'int64' or 'float64'.")
        else:
            self.attn_mask = None
        self.key, self.value = self.query, self.query

        self.dout = np.random.random((self.batch_size, self.query_length,
                                      self.embed_dim)).astype(self.x_type)

    def GetBaselineOut(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        tensor_query = paddle.to_tensor(self.query, stop_gradient=False)

        cache_kv = None
        if self.has_cache_kv:
            cache_kv = paddle.to_tensor(self.cache_kv, stop_gradient=False)

        if self.has_attn_mask:
            attn_mask = paddle.to_tensor(self.attn_mask, stop_gradient=False)
        else:
            attn_mask = None

        for i in range(self.layers):
            residual = tensor_query
            ln1_out = tensor_query
            if self.pre_layer_norm:
                ln1_out = self.norm(tensor_query)

            q = self.q_proj(ln1_out)
            q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
            q_out = tensor.transpose(x=q, perm=[0, 2, 1, 3])
            k = self.k_proj(ln1_out)
            v = self.v_proj(ln1_out)
            k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
            k_out = tensor.transpose(x=k, perm=[0, 2, 1, 3])
            v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
            v_out = tensor.transpose(x=v, perm=[0, 2, 1, 3])

            if self.has_cache_kv:
                # [1, B, n_head, cache_seq_len, head_dim]
                cache_k, cache_v = paddle.split(cache_kv, 2)
                cache_k = paddle.squeeze(cache_k, axis=0)
                cache_v = paddle.squeeze(cache_v, axis=0)
                # [B, n_head, cache_seq_len + seq_len, head_dim]
                # out_seq_len = cache_seq_len + seq_len
                k_out = paddle.concat([cache_k, k_out], axis=-2)
                v_out = paddle.concat([cache_v, v_out], axis=-2)

            # [B, n_head, seq_len, head_dim] * [B, n_head, out_seq_len, head_dim]
            # --> [B, n_head, seq_len, out_seq_len]
            qk_out = layers.matmul(
                x=q_out, y=k_out, transpose_y=True, alpha=self.head_dim**-0.5)

            if attn_mask is not None:
                attn_mask = _convert_attention_mask(attn_mask, qk_out.dtype)
                attn_mask_out = qk_out + attn_mask
                softmax_out = F.softmax(attn_mask_out)
            else:
                softmax_out = F.softmax(qk_out)

            if self.dropout_prob:
                dropout_out = F.dropout(
                    softmax_out,
                    self.dropout_prob,
                    training=self.training,
                    mode="upscale_in_train")
                # [B, n_head, seq_len, out_seq_len] * [B, n_head, out_seq_len, head_dim]
                # --> [B, n_head, seq_len, head_dim]
                qktv_out = tensor.matmul(dropout_out, v_out)
            else:
                qktv_out = tensor.matmul(softmax_out, v_out)

            fmha_out = tensor.transpose(qktv_out, perm=[0, 2, 1, 3])
            out_linear_in = tensor.reshape(
                x=fmha_out,
                shape=[0, 0, fmha_out.shape[2] * fmha_out.shape[3]])
            out = self.out_proj(out_linear_in)

            residual_out = residual + self.dropout(out)
            if not self.pre_layer_norm:
                attn_out = self.norm(residual_out)
            else:
                attn_out = residual_out

            ffn_ln_out = attn_out
            if self.pre_layer_norm:
                ffn_ln_out = self.ffn_norm(attn_out)

            ffn1_out = self.ffn1_proj(ffn_ln_out)
            ffn1_out = self.dropout(self.activation(ffn1_out))
            ffn2_out = self.ffn2_proj(ffn1_out)

            residual_out = attn_out + self.dropout(ffn2_out)
            final_out = residual_out
            if not self.pre_layer_norm:
                final_out = self.ffn_norm(residual_out)

            tensor_query = final_out

        return final_out

    def GetFusedMultiTransformerOut(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        q_proj_weight = paddle.to_tensor(
            self.q_proj.weight, stop_gradient=False)
        k_proj_weight = paddle.to_tensor(
            self.k_proj.weight, stop_gradient=False)
        v_proj_weight = paddle.to_tensor(
            self.v_proj.weight, stop_gradient=False)
        out_linear_weight = paddle.to_tensor(
            self.out_proj.weight, stop_gradient=False)
        ffn1_weight = paddle.to_tensor(
            self.ffn1_proj.weight, stop_gradient=False)
        ffn2_weight = paddle.to_tensor(
            self.ffn2_proj.weight, stop_gradient=False)

        if self.bias_attr is False:
            qkv_bias_tensor = None
            out_linear_bias = None
        else:
            q_proj_bias = paddle.to_tensor(
                self.q_proj.bias, stop_gradient=False)
            k_proj_bias = paddle.to_tensor(
                self.k_proj.bias, stop_gradient=False)
            v_proj_bias = paddle.to_tensor(
                self.v_proj.bias, stop_gradient=False)
            qkv_bias = np.concatenate(
                (q_proj_bias.numpy(), k_proj_bias.numpy(), v_proj_bias.numpy()))
            qkv_bias = qkv_bias.reshape((3, self.num_heads, self.head_dim))
            qkv_bias_tensor = paddle.to_tensor(qkv_bias, stop_gradient=False)
            out_linear_bias = paddle.to_tensor(
                self.out_proj.bias, stop_gradient=False)
            ffn1_bias = paddle.to_tensor(
                self.ffn1_proj.bias, stop_gradient=False)
            ffn2_bias = paddle.to_tensor(
                self.ffn2_proj.bias, stop_gradient=False)

        ln_scale = paddle.to_tensor(self.norm.weight, stop_gradient=False)
        ln_bias = paddle.to_tensor(self.norm.bias, stop_gradient=False)
        ffn_ln_scale = paddle.to_tensor(
            self.ffn_norm.weight, stop_gradient=False)
        ffn_ln_bias = paddle.to_tensor(self.ffn_norm.bias, stop_gradient=False)

        q_proj_weight = q_proj_weight.numpy().transpose((1, 0))
        k_proj_weight = k_proj_weight.numpy().transpose((1, 0))
        v_proj_weight = v_proj_weight.numpy().transpose((1, 0))
        qkv_weight = np.concatenate(
            (q_proj_weight, k_proj_weight, v_proj_weight))
        qkv_weight = qkv_weight.reshape(
            (3, self.num_heads, self.head_dim, self.embed_dim))

        x = paddle.to_tensor(self.query, stop_gradient=False)
        cache_kvs, cache_kv = None, None
        if self.has_cache_kv:
            cache_kvs = []
            cache_kv = paddle.to_tensor(self.cache_kv, stop_gradient=False)
        if self.has_attn_mask:
            attn_mask = paddle.to_tensor(self.attn_mask, stop_gradient=False)
        else:
            attn_mask = None
        qkv_weight_tensor = paddle.to_tensor(qkv_weight, stop_gradient=False)
        epsilon = 1e-05
        ln2_epsilon = 1e-05

        if attn_mask is not None:
            attn_mask = _convert_attention_mask(attn_mask, x.dtype)

        qkv_weights, qkv_biases = [], []
        out_weights, out_biases = [], []
        ln_scales, ln_biases = [], []
        ffn1_weights, ffn1_biases = [], []
        ffn2_weights, ffn2_biases = [], []
        ffn_ln_scales, ffn_ln_biases = [], []
        for i in range(self.layers):
            qkv_weights.append(qkv_weight_tensor)
            qkv_biases.append(qkv_bias_tensor)
            out_weights.append(out_linear_weight)
            out_biases.append(out_linear_bias)
            ln_scales.append(ln_scale)
            ln_biases.append(ln_bias)
            ffn1_weights.append(ffn1_weight)
            ffn1_biases.append(ffn1_bias)
            ffn2_weights.append(ffn2_weight)
            ffn2_biases.append(ffn2_bias)
            ffn_ln_scales.append(ffn_ln_scale)
            ffn_ln_biases.append(ffn_ln_bias)
            if self.has_cache_kv:
                cache_kvs.append(cache_kv)

        final_out = fused_multi_transformer(
            x,
            qkv_weights,
            qkv_biases,
            out_weights,
            out_biases,
            ln_scales,
            ln_biases,
            ffn1_weights,
            ffn1_biases,
            ffn2_weights,
            ffn2_biases,
            ffn_ln_scales,
            ffn_ln_biases,
            pre_layer_norm=self.pre_layer_norm,
            epsilon=epsilon,
            cache_kvs=cache_kvs,
            attn_mask=attn_mask,
            dropout_rate=self.dropout_prob,
            training=self.training)

        if self.has_cache_kv:
            return final_out[0], final_out[1]

        return final_out

    def test_fused_multi_transformer_op(self):
        final_out_ref = self.GetBaselineOut()
        final_out = self.GetFusedMultiTransformerOut()
        print(type(final_out[0]), type(final_out[1]))
        print(len(final_out[1]))
        print(final_out[1])
        np.testing.assert_allclose(
            final_out_ref, final_out.numpy(), rtol=1e-5, atol=1e-4)


# class TestFusedAttentionOpBiasIsNone(TestFusedAttentionOp):
#     def config(self):
#         super().config()
#         self.bias_attr = False
# 
# 
# class TestFusedAttentionOpPreLn(TestFusedAttentionOp):
#     def config(self):
#         super().config()
#         self.pre_layer_norm = True
# 
# 
# class TestFusedAttentionOpNoneAttnMask(TestFusedAttentionOp):
#     def config(self):
#         super().config()
#         self.pre_layer_norm = True
#         self.has_attn_mask = False
# 
# 
# class TestFusedAttentionOpFp16(TestFusedAttentionOp):
#     def config(self):
#         super().config()
#         self.x_type = np.float16
# 
#     def test_fused_attention_op(self):
#         final_out_ref, x_grad_ref = self.GetBaselineOut()
#         final_out, x_grad = self.GetFusedAttentionOut()
#         np.testing.assert_allclose(
#             final_out_ref, final_out.numpy(), rtol=1e-5, atol=1e-1)
#         np.testing.assert_allclose(
#             x_grad_ref, x_grad.numpy(), rtol=1e-5, atol=1e-1)
# 
# 
class TestFusedMultiTransformerOpCacheKV(TestFusedMultiTransformerOp):
    def config(self):
        super().config()
        self.has_cache_kv = True
        self.training = False
        self.query_length = 1
        self.key_length, self.value_length = 1, 1

    def test_fused_multi_transformer_op(self):
        final_out_ref = self.GetBaselineOut()
        final_out, cache_kv_out = self.GetFusedMultiTransformerOut()
        np.testing.assert_allclose(
            final_out_ref, final_out, rtol=1e-5, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
