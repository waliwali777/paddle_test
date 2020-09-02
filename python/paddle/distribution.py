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

# TODO: define the distribution functions 
# __all__ = ['Categorical',
#            'MultivariateNormalDiag',
#            'Normal',
#            'sampling_id',
#            'Uniform']

from __future__ import print_function

from .fluid.layers import control_flow
from .fluid.layers import tensor
from .fluid.layers import ops
from .fluid.layers import nn
from .fluid import core
from .fluid.framework import in_dygraph_mode
from .tensor.math import elementwise_mul, elementwise_div, elementwise_add, elementwise_sub
import math
import numpy as np
import warnings

from .fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype

__all__ = ['Distribution', 'Uniform', 'Normal']


class Distribution(object):
    """
    The abstract base class for probability distributions. Functions are 
    implemented in specific distributions.
    """

    def __init__(self):
        super(Distribution, self).__init__()

    def sample(self):
        """Sampling from the distribution."""
        raise NotImplementedError

    def entropy(self):
        """The entropy of the distribution."""
        raise NotImplementedError

    def kl_divergence(self, other):
        """The KL-divergence between self distributions and other."""
        raise NotImplementedError

    def log_prob(self, value):
        """Log probability density/mass function."""
        raise NotImplementedError

    def probs(self, value):
        """Probability density/mass function."""
        raise NotImplementedError

    def _validate_args(self, *args):
        """
        Argument validation for distribution args
        Args:
            value (float, list, numpy.ndarray, Tensor)
        Raises
            ValueError: if one argument is Tensor, all arguments should be Tensor
        """
        is_variable = False
        is_number = False
        for arg in args:
            if isinstance(arg, tensor.Variable):
                is_variable = True
            else:
                is_number = True

        if is_variable and is_number:
            raise ValueError(
                'if one argument is Tensor, all arguments should be Tensor')

        return is_variable

    def _to_tensor(self, *args):
        """
        Argument convert args to Tensor

        Args:
            value (float, list, numpy.ndarray, Tensor)
        Returns:
            Tensor of args.
        """
        numpy_args = []
        variable_args = []
        tmp = 0.

        for arg in args:
            if isinstance(arg, float):
                arg = [arg]
            elif not isinstance(arg, list) and not isinstance(arg, np.ndarray):
                raise TypeError(
                    "Type of input args must be float, list, numpy.ndarray or Tensor, but received type {}".
                    format(type(arg)))

            arg_np = np.array(arg)
            arg_dtype = arg_np.dtype
            if str(arg_dtype) not in ['float32']:
                # "assign" op doesn't support float64. if dtype is float64, float32 variable will be generated and transformed to float64 later using "cast".
                if str(arg_dtype) not in ['float64']:
                    warnings.warn(
                        "data type of argument only support float32 and float64, your argument will be convert to float32."
                    )
                arg_np = arg_np.astype('float32')
            # tmp is used to support broadcast, it summarizes shape of all the args and get the mixed shape.
            tmp = tmp + arg_np
            numpy_args.append(arg_np)

        dtype = tmp.dtype
        for arg in numpy_args:
            arg_broadcasted, _ = np.broadcast_arrays(arg, tmp)
            arg_variable = tensor.create_tensor(dtype=dtype)
            tensor.assign(arg_broadcasted, arg_variable)
            variable_args.append(arg_variable)

        return tuple(variable_args)


class Uniform(Distribution):
    """Uniform distribution with `low` and `high` parameters.

    Mathematical Details

    The probability density function (pdf) is

    .. math::

        pdf(x; a, b) = \\frac{1}{Z}, \ a <=x <b

    .. math::

        Z = b - a

    In the above equation:

    * :math:`low = a`,
    * :math:`high = b`,
    * :math:`Z`: is the normalizing constant.

    The parameters `low` and `high` must be shaped in a way that supports
    [broadcasting](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/beginners_guide/basic_concept/broadcasting_en.html) (e.g., `high - low` is a valid operation).

    Args:
        low(int|float|list|numpy.ndarray|Tensor): The lower boundary of uniform distribution.The data type is int, float, list, numpy.ndarray or Tensor
        high(int|float|list|numpy.ndarray|Tensor): The higher boundary of uniform distribution.The data type is int, float, list, numpy.ndarray or Tensor
        name(str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Examples:
        .. code-block:: python

          import numpy as np
          import paddle
          from paddle.distribution import Uniform

          paddle.disable_static()
          # Without broadcasting, a single uniform distribution [3, 4]:
          u1 = Uniform(low=3.0, high=4.0)
          # 2 distributions [1, 3], [2, 4]
          u2 = Uniform(low=[1.0, 2.0], high=[3.0, 4.0])
          # 4 distributions
          u3 = Uniform(low=[[1.0, 2.0], [3.0, 4.0]],
                    high=[[1.5, 2.5], [3.5, 4.5]])

          # With broadcasting:
          u4 = Uniform(low=3.0, high=[5.0, 6.0, 7.0])

          # Complete example
          value_npdata = np.array([0.8], dtype="float32")
          value_tensor = paddle.to_tensor(value_npdata)

          uniform = Uniform([0.], [2.])

          sample = uniform.sample([2])
          # a random tensor created by uniform distribution with shape: [2, 1]
          entropy = uniform.entropy()
          # [0.6931472] with shape: [1]
          lp = uniform.log_prob(value_tensor)
          # [-0.6931472] with shape: [1]
          p = uniform.probs(value_tensor)
          # [0.5] with shape: [1]
    """

    def __init__(self, low, high, name=None):
        if not in_dygraph_mode():
            check_type(low, 'low',
                       (int, float, np.ndarray, tensor.Variable, list),
                       'Uniform')
            check_type(high, 'high',
                       (int, float, np.ndarray, tensor.Variable, list),
                       'Uniform')

        self.all_arg_is_float = False
        self.batch_size_unknown = False
        self.name = name if name is not None else 'Uniform'
        self.dtype = 'float32'

        if isinstance(low, int):
            low = float(low)
        if isinstance(high, int):
            high = float(high)

        if self._validate_args(low, high):
            self.batch_size_unknown = True
            self.low = low
            self.high = high
            self.dtype = convert_dtype(low.dtype)
        else:
            if isinstance(low, float) and isinstance(high, float):
                self.all_arg_is_float = True
            if isinstance(
                    low,
                    np.ndarray) and str(low.dtype) in ['float32', 'float64']:
                self.dtype = low.dtype
            elif isinstance(
                    high,
                    np.ndarray) and str(high.dtype) in ['float32', 'float64']:
                self.dtype = high.dtype
            self.low, self.high = self._to_tensor(low, high)
            if self.dtype != convert_dtype(self.low.dtype):
                self.low = tensor.cast(self.low, dtype=self.dtype)
                self.high = tensor.cast(self.high, dtype=self.dtype)

    def sample(self, shape, seed=0):
        """Generate samples of the specified shape.

        Args:
          shape (list): 1D `int32`. Shape of the generated samples.
          seed (int): Python integer number.

        Returns:
          Tensor: A tensor with prepended dimensions shape.The data type is float32.

        """
        if not in_dygraph_mode():
            check_type(shape, 'shape', (list), 'sample')
            check_type(seed, 'seed', (int), 'sample')

        name = self.name + '_sample'
        batch_shape = list((self.low + self.high).shape)
        if self.batch_size_unknown:
            output_shape = shape + batch_shape
            zero_tmp = tensor.fill_constant_batch_size_like(
                self.low + self.high, batch_shape + shape, self.low.dtype, 0.)
            uniform_random_tmp = nn.uniform_random_batch_size_like(
                zero_tmp,
                zero_tmp.shape,
                dtype=convert_dtype(zero_tmp.dtype),
                min=0.,
                max=1.,
                seed=seed)
            zero_tmp_reshape = nn.reshape(zero_tmp, output_shape)
            uniform_random_tmp_reshape = nn.reshape(uniform_random_tmp,
                                                    output_shape)
            output = uniform_random_tmp_reshape * (
                zero_tmp_reshape + self.high - self.low)
            output = elementwise_add(output, self.low, name=name)
            return output
        else:
            output_shape = shape + batch_shape
            output = nn.uniform_random(
                output_shape, seed=seed,
                dtype=convert_dtype(self.low.dtype)) * (tensor.zeros(
                    output_shape, dtype=self.low.dtype) +
                                                        (self.high - self.low))
            output = elementwise_add(output, self.low, name=name)
            if self.all_arg_is_float:
                return nn.reshape(output, shape, name=name)
            else:
                return output

    def log_prob(self, value):
        """Log probability density/mass function.

        Args:
          value (Tensor): The input tensor.

        Returns:
          Tensor: log probability.The data type is same with value.

        """
        name = self.name + '_log_prob'
        if in_dygraph_mode():
            dtype = self.low.dtype
            if value.dtype != dtype and convert_dtype(
                    value.dtype) in ['float32', 'float64']:
                warnings.warn(
                    "dtype of input 'value' needs to be the same as 'low' and 'high'. dtype of 'value' will be converted."
                )
                value = core.ops.cast(value, 'in_dtype', value.dtype,
                                      'out_dtype', dtype)

            # ensure value in [low, high]
            lb_bool = self.low < value
            ub_bool = value < self.high

            lb = core.ops.cast(lb_bool, 'in_dtype', lb_bool.dtype, 'out_dtype',
                               value.dtype)
            ub = core.ops.cast(ub_bool, 'in_dtype', ub_bool.dtype, 'out_dtype',
                               value.dtype)
            return nn.log(lb * ub) - nn.log(self.high - self.low)

        check_variable_and_dtype(value, 'value', ['float32', 'float64'],
                                 'log_prob')
        if value.dtype != self.low.dtype:
            warnings.warn(
                "dtype of input 'value' needs to be the same as 'low' and 'high'. dtype of 'value' will be converted."
            )
            value = tensor.cast(value, dtype=self.low.dtype)

        lb_bool = self.low < value
        ub_bool = value < self.high
        lb = tensor.cast(lb_bool, dtype=value.dtype)
        ub = tensor.cast(ub_bool, dtype=value.dtype)
        return elementwise_sub(
            nn.log(lb * ub), nn.log(self.high - self.low), name=name)

    def probs(self, value):
        """Probability density/mass function.

        Args:
          value (Tensor): The input tensor.

        Returns:
          Tensor: probability.The data type is same with value.

        """
        name = self.name + '_probs'
        if in_dygraph_mode():
            dtype = self.low.dtype
            if value.dtype != dtype and convert_dtype(
                    value.dtype) in ['float32', 'float64']:
                warnings.warn(
                    "dtype of input 'value' needs to be the same as 'low' and 'high'. dtype of 'value' will be converted."
                )
                value = core.ops.cast(value, 'in_dtype', value.dtype,
                                      'out_dtype', dtype)

            lb_bool = self.low < value
            ub_bool = value < self.high

            lb = core.ops.cast(lb_bool, 'in_dtype', lb_bool.dtype, 'out_dtype',
                               value.dtype)
            ub = core.ops.cast(ub_bool, 'in_dtype', ub_bool.dtype, 'out_dtype',
                               value.dtype)
            return (lb * ub) / (self.high - self.low)

        check_variable_and_dtype(value, 'value', ['float32', 'float64'],
                                 'probs')
        if value.dtype != self.low.dtype:
            warnings.warn(
                "dtype of input 'value' needs to be the same as 'low' and 'high'. dtype of 'value' will be converted."
            )
            value = tensor.cast(value, dtype=self.low.dtype)

        lb_bool = self.low < value
        ub_bool = value < self.high
        lb = tensor.cast(lb_bool, dtype=value.dtype)
        ub = tensor.cast(ub_bool, dtype=value.dtype)
        return elementwise_div((lb * ub), (self.high - self.low), name=name)

    def entropy(self):
        """Shannon entropy in nats.

        The entropy is

        .. math::

            entropy(low, high) = \\log (high - low)

        Returns:
          Tensor: Shannon entropy of uniform distribution.The data type is float32.

        """
        name = self.name + '_entropy'
        return nn.log(self.high - self.low, name=name)


class Normal(Distribution):
    """The Normal distribution with location `loc` and `scale` parameters.

    Mathematical details

    The probability density function (pdf) is

    .. math::

        pdf(x; \mu, \sigma) = \\frac{1}{Z}e^{\\frac {-0.5 (x - \mu)^2}  {\sigma^2} }

    .. math::

        Z = (2 \pi \sigma^2)^{0.5}

    In the above equation:

    * :math:`loc = \mu`: is the mean.
    * :math:`scale = \sigma`: is the std.
    * :math:`Z`: is the normalization constant.

    Args:
        loc(int|float|list|numpy.ndarray|Tensor): The mean of normal distribution.The data type is int, float, list, numpy.ndarray or Tensor.
        scale(int|float|list|numpy.ndarray|Tensor): The std of normal distribution.The data type is int, float, list, numpy.ndarray or Tensor.
        name(str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Examples:
        .. code-block:: python
          
          import numpy as np
          import paddle
          from paddle.distribution import Normal

          paddle.disable_static()
          # Define a single scalar Normal distribution.
          dist = Normal(loc=0., scale=3.)
          # Define a batch of two scalar valued Normals.
          # The first has mean 1 and standard deviation 11, the second 2 and 22.
          dist = Normal(loc=[1., 2.], scale=[11., 22.])
          # Get 3 samples, returning a 3 x 2 tensor.
          dist.sample([3])

          # Define a batch of two scalar valued Normals.
          # Both have mean 1, but different standard deviations.
          dist = Normal(loc=1., scale=[11., 22.])

          # Complete example
          value_npdata = np.array([0.8], dtype="float32")
          value_tensor = paddle.to_tensor(value_npdata)

          normal_a = Normal([0.], [1.])
          normal_b = Normal([0.5], [2.])
          sample = normal_a.sample([2])
          # a random tensor created by normal distribution with shape: [2, 1]
          entropy = normal_a.entropy()
          # [1.4189385] with shape: [1]
          lp = normal_a.log_prob(value_tensor)
          # [-1.2389386] with shape: [1]
          p = normal_a.probs(value_tensor)
          # [0.28969154] with shape: [1]
          kl = normal_a.kl_divergence(normal_b)
          # [0.34939718] with shape: [1]
    """

    def __init__(self, loc, scale, name=None):
        if not in_dygraph_mode():
            check_type(loc, 'loc',
                       (int, float, np.ndarray, tensor.Variable, list),
                       'Normal')
            check_type(scale, 'scale',
                       (int, float, np.ndarray, tensor.Variable, list),
                       'Normal')

        self.batch_size_unknown = False
        self.all_arg_is_float = False
        self.name = name if name is not None else 'Normal'
        self.dtype = 'float32'

        if isinstance(loc, int):
            loc = float(loc)
        if isinstance(scale, int):
            scale = float(scale)

        if self._validate_args(loc, scale):
            self.batch_size_unknown = True
            self.loc = loc
            self.scale = scale
            self.dtype = convert_dtype(loc.dtype)
        else:
            if isinstance(loc, float) and isinstance(scale, float):
                self.all_arg_is_float = True
            if isinstance(
                    loc,
                    np.ndarray) and str(loc.dtype) in ['float32', 'float64']:
                self.dtype = loc.dtype
            elif isinstance(
                    scale,
                    np.ndarray) and str(scale.dtype) in ['float32', 'float64']:
                self.dtype = scale.dtype
            self.loc, self.scale = self._to_tensor(loc, scale)
            if self.dtype != convert_dtype(self.loc.dtype):
                self.loc = tensor.cast(self.loc, dtype=self.dtype)
                self.scale = tensor.cast(self.scale, dtype=self.dtype)

    def sample(self, shape, seed=0):
        """Generate samples of the specified shape.

        Args:
          shape (list): 1D `int32`. Shape of the generated samples.
          seed (int): Python integer number.

        Returns:
          Tensor: A tensor with prepended dimensions shape.The data type is float32.

        """
        if not in_dygraph_mode():
            check_type(shape, 'shape', (list), 'sample')
            check_type(seed, 'seed', (int), 'sample')

        batch_shape = list((self.loc + self.scale).shape)
        name = self.name + '_sample'

        if self.batch_size_unknown:
            output_shape = shape + batch_shape
            zero_tmp = tensor.fill_constant_batch_size_like(
                self.loc + self.scale, batch_shape + shape, self.loc.dtype, 0.)
            zero_tmp_reshape = nn.reshape(zero_tmp, output_shape)
            zero_tmp_shape = nn.shape(zero_tmp_reshape)
            normal_random_tmp = nn.gaussian_random(
                zero_tmp_shape,
                mean=0.,
                std=1.,
                seed=seed,
                dtype=convert_dtype(self.loc.dtype))
            output = normal_random_tmp * (zero_tmp_reshape + self.scale)
            output = elementwise_add(output, self.loc, name=name)
            return output
        else:
            output_shape = shape + batch_shape
            output = nn.gaussian_random(output_shape, mean=0., std=1., seed=seed, dtype=convert_dtype(self.loc.dtype)) * \
                     (tensor.zeros(output_shape, dtype=self.loc.dtype) + self.scale)
            output = elementwise_add(output, self.loc, name=name)
            if self.all_arg_is_float:
                return nn.reshape(output, shape, name=name)
            else:
                return output

    def entropy(self):
        """Shannon entropy in nats.

        The entropy is

        .. math::

            entropy(\sigma) = 0.5 \\log (2 \pi e \sigma^2)

        In the above equation:

        * :math:`scale = \sigma`: is the std.

        Returns:
          Tensor: Shannon entropy of normal distribution.The data type is float32.

        """
        name = self.name + '_entropy'
        batch_shape = list((self.loc + self.scale).shape)
        zero_tmp = tensor.fill_constant_batch_size_like(
            self.loc + self.scale, batch_shape, self.loc.dtype, 0.)
        return elementwise_add(
            0.5 + zero_tmp,
            0.5 * math.log(2 * math.pi) + nn.log((self.scale + zero_tmp)),
            name=name)

    def log_prob(self, value):
        """Log probability density/mass function.

        Args:
          value (Tensor): The input tensor.

        Returns:
          Tensor: log probability.The data type is same with value.

        """
        name = self.name + '_log_prob'
        if in_dygraph_mode():
            if value.dtype != self.loc.dtype and convert_dtype(
                    value.dtype) in ['float32', 'float64']:
                warnings.warn(
                    "dtype of input 'value' needs to be the same as 'loc' and 'scale'. dtype of 'value' will be converted."
                )
                value = core.ops.cast(value, 'in_dtype', value.dtype,
                                      'out_dtype', self.loc.dtype)
        else:
            check_variable_and_dtype(value, 'value', ['float32', 'float64'],
                                     'log_prob')
            if value.dtype != self.loc.dtype:
                value = tensor.cast(value, dtype=self.loc.dtype)

        var = self.scale * self.scale
        log_scale = nn.log(self.scale)
        return elementwise_sub(
            -1. * ((value - self.loc) * (value - self.loc)) / (2. * var),
            log_scale + math.log(math.sqrt(2. * math.pi)),
            name=name)

    def probs(self, value):
        """Probability density/mass function.

        Args:
          value (Tensor): The input tensor.

        Returns:
          Tensor: probability.The data type is same with value.

        """
        name = self.name + '_probs'
        if in_dygraph_mode():
            if value.dtype != self.loc.dtype and convert_dtype(
                    value.dtype) in ['float32', 'float64']:
                warnings.warn(
                    "dtype of input 'value' needs to be the same as 'loc' and 'scale'. dtype of 'value' will be converted."
                )
                value = core.ops.cast(value, 'in_dtype', value.dtype,
                                      'out_dtype', self.loc.dtype)
        else:
            check_variable_and_dtype(value, 'value', ['float32', 'float64'],
                                     'probs')
            if value.dtype != self.loc.dtype:
                value = tensor.cast(value, dtype=self.loc.dtype)

        var = self.scale * self.scale
        return elementwise_div(
            ops.exp(-1. * ((value - self.loc) * (value - self.loc)) /
                    (2. * var)), (math.sqrt(2 * math.pi) * self.scale),
            name=name)

    def kl_divergence(self, other):
        """The KL-divergence between two normal distributions.

        The probability density function (pdf) is

        .. math::

            KL\_divergence(\mu_0, \sigma_0, \mu_1, \sigma_1) = 0.5 (ratio^2 + (\\frac{diff}{\sigma_1})^2 - 1 - 2 \\log {ratio})

        .. math::

            ratio = \\frac{\sigma_0}{\sigma_1}
        
        .. math::

            diff = \mu_1 - \mu_0

        In the above equation:

        * :math:`loc = \mu_0`: is the mean of current Normal distribution.
        * :math:`scale = \sigma_0`: is the std of current Normal distribution.
        * :math:`loc = \mu_1`: is the mean of other Normal distribution.
        * :math:`scale = \sigma_1`: is the std of other Normal distribution.
        * :math:`ratio`: is the ratio of scales.
        * :math:`diff`: is the difference between means.

        Args:
            other (Normal): instance of Normal.

        Returns:
            Tensor: kl-divergence between two normal distributions.The data type is float32.

        """
        if not in_dygraph_mode():
            check_type(other, 'other', Normal, 'kl_divergence')

        name = self.name + '_kl_divergence'
        var_ratio = self.scale / other.scale
        var_ratio = (var_ratio * var_ratio)
        t1 = (self.loc - other.loc) / other.scale
        t1 = (t1 * t1)
        return elementwise_add(
            0.5 * var_ratio, 0.5 * (t1 - 1. - nn.log(var_ratio)), name=name)
