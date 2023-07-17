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
from collections import OrderedDict

import numpy as np

import paddle
from paddle.framework import core

alignment = {
    "gpu": 256,
}

align = {
    paddle.float16.value: 2,
    paddle.bfloat16.value: 2,
    paddle.float32.value: 4,
}


def assign_group_by_size(parameters, group_size=256 * 1024 * 1024):
    is_sparse_gradient = [False] * len(parameters)

    group_indices = core.eager_assign_group_by_size(
        parameters, is_sparse_gradient, [group_size, group_size]
    )

    var_groups = OrderedDict()
    for group_idx, indices in enumerate(group_indices):
        for index in indices:
            var_groups.setdefault(group_idx, []).append(parameters[index])
    return var_groups


def flatten_dense_tensors(parameters, use_main_grad):
    from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_storage import (
        GradStorage,
        ParamStorage,
    )

    _buffer_size = 0
    _param2align = {}
    dtype = parameters[0].dtype

    for param in parameters:
        assert param.trainable, "param must be trainable..."
        size = np.prod(param.shape) * align[dtype]
        remaining = size % alignment["gpu"]
        ali = 0 if remaining == 0 else alignment["gpu"] - remaining
        align_ = ali // align[dtype]
        _buffer_size += np.prod(param.shape) + align_
        _param2align[param.name] = align_

    param_storage = ParamStorage(size=_buffer_size, dtype=dtype, device="gpu")

    param_storage.add_rank_params(parameters, _param2align)

    # process gradient
    grad_dtype = paddle.float32 if use_main_grad else dtype
    grad_storage = GradStorage(
        size=_buffer_size,
        dtype=grad_dtype,
        device="gpu",
        destination="0",
        parm2align=_param2align,
    )

    for param in parameters:
        grad_storage.add_grad(param, _param2align[param.name])

    param_storage.warp_buffer()
    grad_storage.warp_buffer()

    if not use_main_grad:
        # param_storage --> grad_storage
        param_storage.buffer._copy_gradient_from(grad_storage.buffer)
    else:
        param_storage.buffer.main_grad = grad_storage.buffer
    param_storage.buffer.stop_gradient = False
    return param_storage, grad_storage


def obtain_storage(parameters, use_main_grad, clip, dist):
    if len(parameters) < 1:
        return []

    var_groups = assign_group_by_size(parameters)
    storage = []
    for group_idx, parameters in var_groups.items():
        param_storage, grad_storage = flatten_dense_tensors(
            parameters, use_main_grad
        )
        param_storage.buffer.need_clip = clip
        param_storage.buffer.is_distributed = dist
        storage.append(param_storage.buffer)
    return storage


def fused_parameters(parameters, use_main_grad):
    # filter for mp's distributed params
    dist = list(filter(lambda x: x.is_distributed, parameters))
    no_dist = list(filter(lambda x: not x.is_distributed, parameters))

    # filter for different dtype
    fp32_dist = list(filter(lambda x: x.dtype == paddle.float32, dist))
    fp32_no_dist = list(filter(lambda x: x.dtype == paddle.float32, no_dist))
    no_fp32_dist = list(filter(lambda x: x.dtype != paddle.float32, dist))
    no_fp32_no_dist = list(filter(lambda x: x.dtype != paddle.float32, no_dist))

    # no fp32 param's dtype should be the same
    all_no_fp32_param = no_fp32_dist + no_fp32_no_dist
    no_fp32_dtype = None
    for p in all_no_fp32_param:
        if no_fp32_dtype is None:
            no_fp32_dtype = p.dtype
        else:
            assert (
                p.dtype == no_fp32_dtype
            ), "Tensor fusion only support two different param dtypes."

    # filter for need clip
    fp32_dist_clip = list(
        filter(lambda x: getattr(x, 'need_clip', True), fp32_dist)
    )
    fp32_no_dist_clip = list(
        filter(lambda x: getattr(x, 'need_clip', True), fp32_no_dist)
    )
    no_fp32_dist_clip = list(
        filter(lambda x: getattr(x, 'need_clip', True), no_fp32_dist)
    )
    no_fp32_no_dist_clip = list(
        filter(lambda x: getattr(x, 'need_clip', True), no_fp32_no_dist)
    )
    fp32_dist_no_clip = list(
        filter(lambda x: not getattr(x, 'need_clip', True), fp32_dist)
    )
    fp32_no_dist_no_clip = list(
        filter(lambda x: not getattr(x, 'need_clip', True), fp32_no_dist)
    )
    no_fp32_dist_no_clip = list(
        filter(lambda x: not getattr(x, 'need_clip', True), no_fp32_dist)
    )
    no_fp32_no_dist_no_clip = list(
        filter(lambda x: not getattr(x, 'need_clip', True), no_fp32_no_dist)
    )

    param_groups = [
        fp32_dist_clip,
        fp32_no_dist_clip,
        no_fp32_dist_clip,
        no_fp32_no_dist_clip,
        fp32_dist_no_clip,
        fp32_no_dist_no_clip,
        no_fp32_dist_no_clip,
        no_fp32_no_dist_no_clip,
    ]
    attrs = [
        [paddle.float32, True, True],
        [paddle.float32, False, True],
        [no_fp32_dtype, True, True],
        [no_fp32_dtype, False, True],
        [paddle.float32, True, False],
        [paddle.float32, False, False],
        [no_fp32_dtype, True, False],
        [no_fp32_dtype, False, False],
    ]

    decay_fused = []
    all_fused = []
    for params, attr in zip(param_groups, attrs):
        decay_params = []
        other_params = []

        for param in params:
            if not any(nd in param.name for nd in ["bias", "norm", "b_0"]):
                decay_params.append(param)
            else:
                other_params.append(param)

        decay = obtain_storage(decay_params, use_main_grad, attr[2], attr[1])
        other = obtain_storage(other_params, use_main_grad, attr[2], attr[1])
        decay_fused += decay
        all_fused += decay
        all_fused += other

    return decay_fused, all_fused
