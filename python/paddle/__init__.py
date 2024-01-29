# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
try:
    from paddle.cuda_env import *  # noqa: F403
    from paddle.version import (  # noqa: F401
        commit as __git_commit__,
        full_version as __version__,
    )
except ImportError:
    import sys

    sys.stderr.write(
        '''Warning with import paddle: you should not
     import paddle from the source directory; please install paddlepaddle*.whl firstly.'''
    )

# NOTE(SigureMo): We should place the import of base.core before other modules,
# because there are some initialization codes in base/core/__init__.py.
from .base import core  # noqa: F401
from .batch import batch

# Do the *DUPLICATED* monkey-patch for the tensor object.
# We need remove the duplicated code here once we fix
# the illogical implement in the monkey-patch methods later.
from .framework import monkey_patch_math_tensor, monkey_patch_variable
from .pir import monkey_patch_program, monkey_patch_value

monkey_patch_variable()
monkey_patch_math_tensor()
monkey_patch_value()
monkey_patch_program()

from .base.dataset import *  # noqa: F403
from .framework import (
    disable_signal_handler,
    disable_static,
    enable_static,
    get_flags,
    in_dynamic_mode,
    set_flags,
)
from .framework.dtype import (
    bfloat16,
    bool,
    complex64,
    complex128,
    dtype,
    finfo,
    float16,
    float32,
    float64,
    iinfo,
    int8,
    int16,
    int32,
    int64,
    uint8,
)

Tensor = framework.core.eager.Tensor
Tensor.__qualname__ = 'Tensor'

import paddle.distributed.fleet
import paddle.text
import paddle.vision  # noqa: F401
from paddle import (  # noqa: F401
    amp,
    audio,
    autograd,
    dataset,
    decomposition,
    device,
    distributed,
    distribution,
    geometric,
    incubate,
    inference,
    io,
    jit,
    metric,
    nn,
    onnx,
    optimizer,
    quantization,
    reader,
    regularizer,
    sparse,
    static,
    sysconfig,
    vision,
)

# high-level api
from . import (  # noqa: F401
    _pir_ops,
    callbacks,
    fft,
    hub,
    linalg,
    signal,
)
from .autograd import (
    enable_grad,
    grad,
    is_grad_enabled,
    no_grad,
    set_grad_enabled,
)
from .device import (  # noqa: F401
    get_cudnn_version,
    get_device,
    is_compiled_with_cinn,
    is_compiled_with_cuda,
    is_compiled_with_custom_device,
    is_compiled_with_distribute,
    is_compiled_with_ipu,
    is_compiled_with_rocm,
    is_compiled_with_xpu,
    set_device,
)
from .distributed import DataParallel
from .framework import (  # noqa: F401
    CPUPlace,
    CUDAPinnedPlace,
    CUDAPlace,
    CustomPlace,
    IPUPlace,
    ParamAttr,
    XPUPlace,
    async_save,
    clear_async_save_task_queue,
    get_default_dtype,
    load,
    save,
    set_default_dtype,
)
from .framework.random import (
    get_cuda_rng_state,
    get_rng_state,
    seed,
    set_cuda_rng_state,
    set_rng_state,
)
from .hapi import (
    Model,
    flops,
    summary,
)
from .nn.functional.distance import (
    pdist,
)
from .nn.initializer.lazy_init import LazyGuard
from .tensor.attribute import (
    imag,
    is_complex,
    is_floating_point,
    is_integer,
    rank,
    real,
    shape,
)
from .tensor.creation import (
    arange,
    assign,
    cauchy_,
    clone,
    complex,
    create_parameter,
    diag,
    diag_embed,
    diagflat,
    empty,
    empty_like,
    eye,
    full,
    full_like,
    geometric_,
    linspace,
    logspace,
    meshgrid,
    ones,
    ones_like,
    polar,
    to_tensor,
    tril,
    tril_,
    tril_indices,
    triu,
    triu_,
    triu_indices,
    zeros,
    zeros_like,
)
from .tensor.einsum import einsum
from .tensor.linalg import (  # noqa: F401
    bincount,
    bmm,
    cdist,
    cholesky,
    cross,
    dist,
    dot,
    eigvalsh,
    histogram,
    histogramdd,
    matmul,
    mv,
    norm,
    t,
    t_,
    transpose,
    transpose_,
)
from .tensor.logic import (
    allclose,
    bitwise_and,
    bitwise_and_,
    bitwise_not,
    bitwise_not_,
    bitwise_or,
    bitwise_or_,
    bitwise_xor,
    bitwise_xor_,
    equal,
    equal_,
    equal_all,
    greater_equal,
    greater_equal_,
    greater_than,
    greater_than_,
    is_empty,
    is_tensor,
    isclose,
    less_equal,
    less_equal_,
    less_than,
    less_than_,
    logical_and,
    logical_and_,
    logical_not,
    logical_not_,
    logical_or,
    logical_or_,
    logical_xor,
    logical_xor_,  # noqa: F401
    not_equal,
    not_equal_,  # noqa: F401
)
from .tensor.manipulation import (
    as_complex,
    as_real,
    as_strided,
    atleast_1d,
    atleast_2d,
    atleast_3d,
    broadcast_tensors,
    broadcast_to,
    cast,
    cast_,
    chunk,
    column_stack,
    concat,
    crop,
    diagonal_scatter,
    dsplit,
    dstack,
    expand,
    expand_as,
    flatten,
    flip,
    flip as reverse,
    gather,
    gather_nd,
    hsplit,
    hstack,
    index_add,
    index_add_,
    index_fill,
    index_fill_,
    index_put,
    index_put_,
    masked_fill,
    masked_fill_,
    masked_scatter,
    masked_scatter_,
    moveaxis,
    put_along_axis,
    repeat_interleave,
    reshape,
    reshape_,
    roll,
    rot90,
    row_stack,
    scatter,
    scatter_,
    scatter_nd,
    scatter_nd_add,
    select_scatter,
    shard_index,
    slice,
    slice_scatter,
    split,
    squeeze,
    squeeze_,
    stack,
    strided_slice,
    take_along_axis,
    tensor_split,
    tensordot,
    tile,
    tolist,
    unbind,
    unflatten,
    unfold,
    unique,
    unique_consecutive,
    unsqueeze,
    unsqueeze_,
    unstack,
    view,
    view_as,
    vsplit,
    vstack,
)
from .tensor.math import (  # noqa: F401
    abs,
    abs_,
    acos,
    acos_,
    acosh,
    acosh_,
    add,
    add_n,
    addmm,
    addmm_,
    all,
    amax,
    amin,
    angle,
    any,
    asin,
    asin_,
    asinh,
    asinh_,
    atan,
    atan2,
    atan_,
    atanh,
    atanh_,
    bitwise_left_shift,
    bitwise_left_shift_,
    bitwise_right_shift,
    bitwise_right_shift_,
    broadcast_shape,
    ceil,
    clip,
    combinations,
    conj,
    copysign,
    copysign_,
    cos,
    cos_,
    cosh,
    cosh_,
    count_nonzero,
    cummax,
    cummin,
    cumprod,
    cumprod_,
    cumsum,
    cumsum_,
    cumulative_trapezoid,
    deg2rad,
    diagonal,
    diff,
    digamma,
    digamma_,
    divide,
    divide_,
    erf,
    erf_,
    erfinv,
    exp,
    expm1,
    expm1_,
    floor,
    floor_divide,
    floor_divide_,
    floor_mod,
    floor_mod_,
    fmax,
    fmin,
    frac,
    frac_,
    frexp,
    gammainc,
    gammainc_,
    gammaincc,
    gammaincc_,
    gammaln,
    gammaln_,
    gcd,
    gcd_,
    heaviside,
    hypot,
    hypot_,
    i0,
    i0_,
    i0e,
    i1,
    i1e,
    increment,
    inner,
    inverse,
    isfinite,
    isinf,
    isnan,
    kron,
    lcm,
    lcm_,
    ldexp,
    ldexp_,
    lerp,
    lgamma,
    lgamma_,
    log,
    log1p,
    log1p_,
    log2,
    log2_,
    log10,
    log10_,
    log_,
    logaddexp,
    logcumsumexp,
    logit,
    logit_,
    logsumexp,
    max,
    maximum,
    min,
    minimum,
    mm,
    mod,
    mod_,
    multigammaln,
    multigammaln_,
    multiplex,
    multiply,
    multiply_,
    nan_to_num,
    nan_to_num_,
    nanmean,
    nansum,
    neg,
    neg_,
    nextafter,
    outer,
    polygamma,
    polygamma_,
    pow,
    pow_,
    prod,
    rad2deg,
    reciprocal,
    remainder,
    remainder_,
    renorm,
    renorm_,
    round,
    rsqrt,
    scale,
    sgn,
    sign,
    signbit,
    sin,
    sin_,
    sinh,
    sinh_,
    sqrt,
    square,
    square_,
    stanh,
    subtract,
    sum,
    take,
    tan,
    tan_,
    tanh,
    tanh_,
    trace,
    trapezoid,
    trunc,
    trunc_,
    vander,
)
from .tensor.random import (
    bernoulli,
    binomial,
    check_shape,
    multinomial,
    normal,
    normal_,
    poisson,
    rand,
    randint,
    randint_like,
    randn,
    randperm,
    standard_gamma,
    standard_normal,
    uniform,
)
from .tensor.search import (
    argmax,
    argmin,
    argsort,
    bucketize,
    index_sample,
    index_select,
    kthvalue,
    masked_select,
    mode,
    nonzero,
    searchsorted,
    sort,
    topk,
    where,
    where_,
)
from .tensor.stat import (
    mean,
    median,
    nanmedian,
    nanquantile,
    numel,
    quantile,
    std,
    var,
)
from .tensor.to_string import set_printoptions

# CINN has to set a flag to include a lib
if is_compiled_with_cinn():
    import os

    package_dir = os.path.dirname(os.path.abspath(__file__))
    runtime_include_dir = os.path.join(package_dir, "libs")
    cuh_file = os.path.join(runtime_include_dir, "cinn_cuda_runtime_source.cuh")
    if os.path.exists(cuh_file):
        os.environ.setdefault('runtime_include_dir', runtime_include_dir)

if is_compiled_with_cuda():
    import os

    package_dir = os.path.dirname(os.path.abspath(__file__))
    cublas_lib_path = package_dir + "/.." + "/nvidia/cublas/lib"
    set_flags({"FLAGS_cublas_dir": cublas_lib_path})

    cudnn_lib_path = package_dir + "/.." + "/nvidia/cudnn/lib"
    set_flags({"FLAGS_cudnn_dir": cudnn_lib_path})

    curand_lib_path = package_dir + "/.." + "/nvidia/curand/lib"
    set_flags({"FLAGS_curand_dir": curand_lib_path})

    cusolver_lib_path = package_dir + "/.." + "/nvidia/cusolver/lib"
    set_flags({"FLAGS_cusolver_dir": cusolver_lib_path})

    cusparse_lib_path = package_dir + "/.." + "/nvidia/cusparse/lib"
    set_flags({"FLAGS_cusparse_dir": cusparse_lib_path})

    nccl_lib_path = package_dir + "/.." + "/nvidia/nccl/lib"
    set_flags({"FLAGS_nccl_dir": nccl_lib_path})

    cupti_dir_lib_path = package_dir + "/.." + "/nvidia/cuda_cupti/lib"
    set_flags({"FLAGS_cupti_dir": cupti_dir_lib_path})


disable_static()

from .pir_utils import IrGuard

ir_guard = IrGuard()
ir_guard._switch_to_pir()

__all__ = [
    'iinfo',
    'finfo',
    'dtype',
    'uint8',
    'int8',
    'int16',
    'int32',
    'int64',
    'float16',
    'float32',
    'float64',
    'bfloat16',
    'bool',
    'complex64',
    'complex128',
    'addmm',
    'addmm_',
    'allclose',
    'isclose',
    't',
    't_',
    'add',
    'subtract',
    'diag',
    'diagflat',
    'diag_embed',
    'isnan',
    'scatter_nd_add',
    'unstack',
    'get_default_dtype',
    'save',
    'multinomial',
    'get_cuda_rng_state',
    'get_rng_state',
    'rank',
    'empty_like',
    'eye',
    'cumsum',
    'cumsum_',
    'cummax',
    'cummin',
    'cumprod',
    'cumprod_',
    'logaddexp',
    'logcumsumexp',
    'logit',
    'logit_',
    'LazyGuard',
    'sign',
    'is_empty',
    'equal',
    'equal_',
    'equal_all',
    'is_tensor',
    'is_complex',
    'is_integer',
    'cross',
    'where',
    'where_',
    'log1p',
    'cos',
    'cos_',
    'tan',
    'tan_',
    'mean',
    'mode',
    'mv',
    'in_dynamic_mode',
    'min',
    'amin',
    'any',
    'slice',
    'slice_scatter',
    'normal',
    'normal_',
    'logsumexp',
    'full',
    'unsqueeze',
    'unsqueeze_',
    'argmax',
    'Model',
    'summary',
    'flops',
    'sort',
    'searchsorted',
    'bucketize',
    'split',
    'tensor_split',
    'hsplit',
    'dsplit',
    'vsplit',
    'logical_and',
    'logical_and_',
    'full_like',
    'less_than',
    'less_than_',
    'kron',
    'clip',
    'Tensor',
    'crop',
    'ParamAttr',
    'stanh',
    'randint',
    'randint_like',
    'assign',
    'gather',
    'scale',
    'zeros',
    'rsqrt',
    'squeeze',
    'squeeze_',
    'to_tensor',
    'gather_nd',
    'isinf',
    'uniform',
    'floor_divide',
    'floor_divide_',
    'remainder',
    'remainder_',
    'floor_mod',
    'floor_mod_',
    'roll',
    'batch',
    'max',
    'amax',
    'logical_or',
    'logical_or_',
    'bitwise_and',
    'bitwise_and_',
    'bitwise_or',
    'bitwise_or_',
    'bitwise_xor',
    'bitwise_xor_',
    'bitwise_not',
    'bitwise_not_',
    'mm',
    'flip',
    'rot90',
    'bincount',
    'histogram',
    'histogramdd',
    'multiplex',
    'CUDAPlace',
    'empty',
    'shape',
    'real',
    'imag',
    'is_floating_point',
    'complex',
    'reciprocal',
    'rand',
    'less_equal',
    'less_equal_',
    'triu',
    'triu_',
    'sin',
    'sin_',
    'dist',
    'cdist',
    'pdist',
    'unbind',
    'meshgrid',
    'arange',
    'load',
    'numel',
    'median',
    'nanmedian',
    'quantile',
    'nanquantile',
    'no_grad',
    'enable_grad',
    'set_grad_enabled',
    'is_grad_enabled',
    'mod',
    'mod_',
    'abs',
    'abs_',
    'tril',
    'tril_',
    'pow',
    'pow_',
    'zeros_like',
    'maximum',
    'topk',
    'index_select',
    'CPUPlace',
    'matmul',
    'seed',
    'acos',
    'acos_',
    'logical_xor',
    'exp',
    'expm1',
    'expm1_',
    'bernoulli',
    'binomial',
    'poisson',
    'standard_gamma',
    'sinh',
    'sinh_',
    'round',
    'DataParallel',
    'argmin',
    'prod',
    'broadcast_shape',
    'conj',
    'neg',
    'neg_',
    'lgamma',
    'lgamma_',
    'gammaincc',
    'gammaincc_',
    'gammainc',
    'gammainc_',
    'lerp',
    'erfinv',
    'inner',
    'outer',
    'square',
    'square_',
    'divide',
    'divide_',
    'gammaln',
    'gammaln_',
    'ceil',
    'atan',
    'atan_',
    'atan2',
    'rad2deg',
    'deg2rad',
    'gcd',
    'gcd_',
    'lcm',
    'lcm_',
    'expand',
    'broadcast_to',
    'ones_like',
    'index_sample',
    'cast',
    'cast_',
    'grad',
    'all',
    'ones',
    'not_equal',
    'sum',
    'nansum',
    'nanmean',
    'count_nonzero',
    'tile',
    'greater_equal',
    'greater_equal_',
    'isfinite',
    'create_parameter',
    'dot',
    'increment',
    'erf',
    'erf_',
    'bmm',
    'chunk',
    'tolist',
    'tensordot',
    'greater_than',
    'greater_than_',
    'shard_index',
    'argsort',
    'tanh',
    'tanh_',
    'transpose',
    'transpose_',
    'cauchy_',
    'geometric_',
    'randn',
    'strided_slice',
    'unique',
    'unique_consecutive',
    'set_cuda_rng_state',
    'set_rng_state',
    'set_printoptions',
    'std',
    'flatten',
    'asin',
    'multiply',
    'multiply_',
    'disable_static',
    'masked_select',
    'var',
    'trace',
    'enable_static',
    'scatter_nd',
    'set_default_dtype',
    'disable_signal_handler',
    'expand_as',
    'stack',
    'hstack',
    'vstack',
    'dstack',
    'column_stack',
    'row_stack',
    'sqrt',
    'randperm',
    'linspace',
    'logspace',
    'reshape',
    'reshape_',
    'atleast_1d',
    'atleast_2d',
    'atleast_3d',
    'reverse',
    'nonzero',
    'CUDAPinnedPlace',
    'logical_not',
    'logical_not_',
    'add_n',
    'minimum',
    'scatter',
    'scatter_',
    'floor',
    'cosh',
    'log',
    'log_',
    'log2',
    'log2_',
    'log10',
    'log10_',
    'concat',
    'check_shape',
    'trunc',
    'trunc_',
    'frac',
    'frac_',
    'digamma',
    'digamma_',
    'standard_normal',
    'diagonal',
    'broadcast_tensors',
    'einsum',
    'set_flags',
    'get_flags',
    'asinh',
    'acosh',
    'atanh',
    'as_complex',
    'as_real',
    'diff',
    'angle',
    'fmax',
    'fmin',
    'moveaxis',
    'repeat_interleave',
    'clone',
    'kthvalue',
    'renorm',
    'renorm_',
    'take_along_axis',
    'put_along_axis',
    'select_scatter',
    'multigammaln',
    'multigammaln_',
    'nan_to_num',
    'nan_to_num_',
    'heaviside',
    'tril_indices',
    'index_add',
    "index_add_",
    "index_put",
    "index_put_",
    'sgn',
    'triu_indices',
    'take',
    'frexp',
    'ldexp',
    'ldexp_',
    'trapezoid',
    'cumulative_trapezoid',
    'polar',
    'vander',
    'unflatten',
    'as_strided',
    'view',
    'view_as',
    'unfold',
    'nextafter',
    'i0',
    'i0_',
    'i0e',
    'i1',
    'i1e',
    'polygamma',
    'polygamma_',
    'copysign',
    'copysign_',
    'bitwise_left_shift',
    'bitwise_left_shift_',
    'bitwise_right_shift',
    'bitwise_right_shift_',
    'masked_fill',
    'masked_fill_',
    'masked_scatter',
    'masked_scatter_',
    'hypot',
    'hypot_',
    'index_fill',
    "index_fill_",
    'diagonal_scatter',
    'combinations',
    'signbit',
]
