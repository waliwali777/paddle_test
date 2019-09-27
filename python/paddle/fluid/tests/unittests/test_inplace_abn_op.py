#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
import os
import six
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid import compiler
import paddle.fluid.unique_name as unique_name


class TestInplaceANBOpTraining(unittest.TestCase):
    def setUp(self):
        self.dtype = np.float64
        self.N = 32
        self.C = 16
        self.H = 64
        self.W = 32
        self.dshape = [self.N, self.C, self.H, self.W]

    def build_program(self,
                      place,
                      layout,
                      seed,
                      only_forward=False,
                      activation="identity",
                      sync_bn=False,
                      use_cuda=False):
        main = fluid.Program()
        startup = fluid.Program()
        main.random_seed = seed
        startup.random_seed = seed
        with fluid.unique_name.guard():
            with fluid.program_guard(main, startup):
                data = fluid.layers.data(
                    name='input',
                    shape=self.dshape,
                    dtype=self.dtype,
                    append_batch_size=False)
                conv = fluid.layers.conv2d(
                    input=data,
                    num_filters=32,
                    filter_size=1,
                    param_attr=fluid.ParamAttr(name='conv2d_weight'),
                    bias_attr=False,
                    use_cudnn=False)
                bn = fluid.layers.batch_norm(
                    conv,
                    act=activation,
                    param_attr=fluid.ParamAttr(name='bn_scale'),
                    bias_attr=fluid.ParamAttr(name='bn_bias'),
                    moving_mean_name='bn_moving_mean',
                    moving_variance_name='bn_moving_variance',
                    data_layout=layout,
                    is_test=only_forward)

                sigmoid = fluid.layers.sigmoid(bn)
                out = fluid.layers.reduce_sum(sigmoid)
                if use_cuda and not sync_bn:
                    out = out / core.get_cuda_device_count()
                if not only_forward:
                    sgd_opt = fluid.optimizer.SGD(learning_rate=0.0)
                    sgd_opt.backward(out)
        return main, startup, [out, conv]

    def compare(self, place, layout, only_forward, activation, use_cuda):
        seed = 10
        os.environ['FLAGS_cudnn_deterministic'] = "1"
        data = np.random.random(size=self.dshape).astype(self.dtype) * 4. - 2
        # Single-GPU, N = 32 per GPU
        main, startup, outs = self.build_program(
            place, layout, seed, only_forward, activation, False, use_cuda)
        exe = fluid.Executor(place)
        exe.run(startup)
        fetch_names = [v.name for v in outs] + [
            'bn_moving_mean', 'bn_moving_variance', 'bn_scale', 'bn_bias'
        ]
        if not only_forward:
            others = [
                'batch_norm_0.tmp_0',
                'batch_norm_0.tmp_1',
                'bn_scale@GRAD',
                'bn_bias@GRAD',
                # 'batch_norm_0.tmp_2@GRAD',
                'conv2d_0.tmp_0@GRAD'
            ]
            fetch_names += others
        for nm in fetch_names:
            fv = fluid.framework._get_var(str(nm), program=main)
            fv.persistable = True
        bn_fetches = exe.run(program=main,
                             feed={'input': data},
                             fetch_list=fetch_names)

        #####################################################################
        # Multi-GPUs, self.N / core.get_cuda_device_count() per GPU
        main, startup, outs = self.build_program(
            place, layout, seed, only_forward, activation, True, use_cuda)
        exe = fluid.Executor(place)
        exe.run(startup)
        fetch_names1 = [v.name for v in outs] + [
            'bn_moving_mean', 'bn_moving_variance', 'bn_scale', 'bn_bias'
        ]

        if not only_forward:
            fetch_names1 += others
        for nm in fetch_names1:
            fv = fluid.framework._get_var(str(nm), program=main)
            fv.persistable = True
        build_strategy = fluid.BuildStrategy()
        build_strategy.sync_batch_norm = use_cuda
        build_strategy.enable_inplace = True
        build_strategy.memory_optimize = False
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = 1 if os.name == 'nt' else 0
        comp_prog = compiler.CompiledProgram(main).with_data_parallel(
            outs[0].name if not only_forward else None,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)
        inplace_abn_fetches = exe.run(program=comp_prog,
                                      feed={'input': data},
                                      fetch_list=fetch_names1)

        for i in six.moves.xrange(1, len(inplace_abn_fetches)):
            bn_val = bn_fetches[i]
            inplace_abn_val = inplace_abn_fetches[i]
            if inplace_abn_val.shape != bn_val.shape:
                inplace_abn_val = inplace_abn_val[:bn_val.shape[0]]
            self.assertTrue(
                np.allclose(
                    bn_val, inplace_abn_val, atol=1e-100),
                "Output (" + fetch_names[i] + ":" + fetch_names1[i] +
                ") has diff. \n" + "\nBN     " + str(bn_val) + "\n" +
                "Inplace ABN " + str(inplace_abn_val))

    def test_op(self):
        use_cudas = [False]
        if core.is_compiled_with_cuda():
            use_cudas.append(True)

        for use_cuda in use_cudas:
            place = core.CUDAPlace(0) if use_cuda else core.CPUPlace()
            layouts = ["NCHW", "NHWC"] if use_cuda else [
                "NCHW"
            ]  #NHWC can be too slow under cpu mode
            for layout in layouts:
                for activation in ['elu', 'relu']:
                    for infer_only in [False, True]:
                        self.compare(place, layout, infer_only, activation,
                                     use_cuda)


if __name__ == '__main__':
    unittest.main()
