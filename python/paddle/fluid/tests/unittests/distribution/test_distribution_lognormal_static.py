# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import paddle
import numpy as np
import scipy.stats
import config

from parameterize import TEST_CASE_NAME, parameterize_cls, place, xrand
from paddle.distribution.normal import Normal
from paddle.distribution.lognormal import LogNormal
from test_distribution_lognormal import LogNormalNumpy
from paddle.distribution.kl import kl_divergence

np.random.seed(2022)


@place(config.DEVICES)
@parameterize_cls((TEST_CASE_NAME, 'loc', 'scale'), [('one-dim', xrand(
    (2, )), xrand((2, ))), ('multi-dim', xrand((3, 3)), xrand((3, 3)))])
class TestLogNormal(unittest.TestCase):

    def setUp(self):
        paddle.enable_static()
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(main_program, startup_program):
            loc = paddle.static.data('loc', self.loc.shape, self.loc.dtype)
            scale = paddle.static.data('scale', self.scale.shape,
                                       self.scale.dtype)
            self.ln_a = LogNormal(loc=loc, scale=scale)
            self._np_lognormal = LogNormalNumpy(loc=self.loc, scale=self.scale)
            mean = self.ln_a.mean
            var = self.ln_a.variance
            entropy = self.ln_a.entropy()
        fetch_list = [mean, var, entropy]
        self.feeds = {'loc': self.loc, 'scale': self.scale}

        executor.run(startup_program)
        [self.mean, self.var,
         self.entropy] = executor.run(main_program,
                                      feed=self.feeds,
                                      fetch_list=fetch_list)

    def test_mean(self):
        np_mean = self._np_lognormal.mean
        self.assertEqual(str(self.mean.dtype).split('.')[-1], self.scale.dtype)
        np.testing.assert_allclose(self.mean,
                                   np_mean,
                                   rtol=config.RTOL.get(str(self.scale.dtype)),
                                   atol=config.ATOL.get(str(self.scale.dtype)))

    def test_var(self):
        np_var = self._np_lognormal.variance
        self.assertEqual(str(self.var.dtype).split('.')[-1], self.scale.dtype)
        np.testing.assert_allclose(self.var,
                                   np_var,
                                   rtol=config.RTOL.get(str(self.scale.dtype)),
                                   atol=config.ATOL.get(str(self.scale.dtype)))

    def test_entropy(self):
        np_entropy = self._np_lognormal.entropy()
        self.assertEqual(
            str(self.entropy.dtype).split('.')[-1], self.scale.dtype)
        np.testing.assert_allclose(self.entropy,
                                   np_entropy,
                                   rtol=config.RTOL.get(str(self.scale.dtype)),
                                   atol=config.ATOL.get(str(self.scale.dtype)))


@place(config.DEVICES)
@parameterize_cls((TEST_CASE_NAME, 'loc', 'scale'),
                  [('sample', xrand(
                      (4, ), min=0, max=1), xrand((4, ), min=0.01, max=1))])
class TestLogNormalSample(unittest.TestCase):

    def setUp(self):
        paddle.enable_static()
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(main_program, startup_program):
            loc = paddle.static.data('loc', self.loc.shape, self.loc.dtype)
            scale = paddle.static.data('scale', self.scale.shape,
                                       self.scale.dtype)
            n = 80000
            self.sample_shape = [n]
            self.rsample_shape = [n]
            self.ln_a = LogNormal(loc=loc, scale=scale)
            self.mean = self.ln_a.mean
            self.variance = self.ln_a.variance
            self.samples = self.ln_a.sample(self.sample_shape)
            self.rsamples = self.ln_a.rsample(self.rsample_shape)
        fetch_list = [self.mean, self.variance, self.samples, self.rsamples]
        self.feeds = {'loc': self.loc, 'scale': self.scale}

        executor.run(startup_program)
        [self.mean, self.variance, self.samples,
         self.rsamples] = executor.run(main_program,
                                       feed=self.feeds,
                                       fetch_list=fetch_list)

    def test_sample(self):
        samples_mean = self.samples.mean(axis=0)
        samples_var = self.samples.var(axis=0)
        np.testing.assert_allclose(samples_mean, self.mean, rtol=0.1, atol=0)
        np.testing.assert_allclose(samples_var, self.variance, rtol=0.1, atol=0)

        rsamples_mean = self.rsamples.mean(axis=0)
        rsamples_var = self.rsamples.var(axis=0)
        np.testing.assert_allclose(rsamples_mean, self.mean, rtol=0.1, atol=0)
        np.testing.assert_allclose(rsamples_var,
                                   self.variance,
                                   rtol=0.1,
                                   atol=0)

        for i in range(len(self.scale)):
            self.assertEqual(self.samples[:, i].shape, tuple(self.sample_shape))
            self.assertEqual(self.rsamples[:, i].shape,
                             tuple(self.rsample_shape))
            self.assertTrue(
                self._kstest(self.loc[i], self.scale[i], self.samples[:, i]))
            self.assertTrue(
                self._kstest(self.loc[i], self.scale[i], self.rsamples[:, i]))

    def _kstest(self, loc, scale, samples):
        # Uses the Kolmogorov-Smirnov test for goodness of fit.
        ks, _ = scipy.stats.kstest(
            samples,
            scipy.stats.lognorm(s=scale, scale=np.exp(loc)).cdf)
        return ks < 0.02


@place(config.DEVICES)
@parameterize_cls(
    (TEST_CASE_NAME, 'loc1', 'scale1', 'loc2', 'scale2'),
    [('one-dim', xrand((2, )), xrand((2, )), xrand((2, )), xrand((2, ))),
     ('multi-dim', xrand((2, 2)), xrand((2, 2)), xrand((2, 2)), xrand((2, 2)))])
class TestLogNormalKL(unittest.TestCase):

    def setUp(self):
        paddle.enable_static()
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(main_program, startup_program):
            loc1 = paddle.static.data('loc1', self.loc1.shape, self.loc1.dtype)
            scale1 = paddle.static.data('scale1', self.scale1.shape,
                                        self.scale1.dtype)
            loc2 = paddle.static.data('loc2', self.loc2.shape, self.loc2.dtype)
            scale2 = paddle.static.data('scale2', self.scale2.shape,
                                        self.scale2.dtype)

            self.ln_a = LogNormal(loc=loc1, scale=scale1)
            self.ln_b = LogNormal(loc=loc2, scale=scale2)
            self.normal_a = Normal(loc=loc1, scale=scale1)
            self.normal_b = Normal(loc=loc2, scale=scale2)

            self.kl0 = self.ln_a.kl_divergence(self.ln_b)
            self.kl1 = kl_divergence(self.ln_a, self.ln_b)
            self.kl_normal = kl_divergence(self.normal_a, self.normal_b)
            self.kl_formula = self._kl(self.ln_a, self.ln_b)

        fetch_list = [self.kl0, self.kl1, self.kl_normal, self.kl_formula]
        self.feeds = {
            'loc1': self.loc1,
            'scale1': self.scale1,
            'loc2': self.loc2,
            'scale2': self.scale2
        }

        executor.run(startup_program)
        [self.kl0, self.kl1, self.kl_normal,
         self.kl_formula] = executor.run(main_program,
                                         feed=self.feeds,
                                         fetch_list=fetch_list)

    def test_kl_divergence(self):
        np.testing.assert_allclose(self.kl0,
                                   self.kl_formula,
                                   rtol=config.RTOL.get(str(self.scale1.dtype)),
                                   atol=config.ATOL.get(str(self.scale1.dtype)))

        np.testing.assert_allclose(self.kl1,
                                   self.kl_formula,
                                   rtol=config.RTOL.get(str(self.scale1.dtype)),
                                   atol=config.ATOL.get(str(self.scale1.dtype)))

        np.testing.assert_allclose(self.kl_normal,
                                   self.kl_formula,
                                   rtol=config.RTOL.get(str(self.scale1.dtype)),
                                   atol=config.ATOL.get(str(self.scale1.dtype)))

    def _kl(self, dist1, dist2):
        loc1 = dist1.loc
        loc2 = dist2.loc
        scale1 = (dist1.scale)
        scale2 = (dist2.scale)
        var_ratio = (scale1 / scale2)
        var_ratio = var_ratio * var_ratio
        t1 = ((loc1 - loc2) / scale2)
        t1 = (t1 * t1)
        return 0.5 * (var_ratio + t1 - 1 - np.log(var_ratio))


if __name__ == '__main__':
    unittest.main()
