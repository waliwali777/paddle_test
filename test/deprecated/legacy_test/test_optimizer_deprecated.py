#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import framework
from paddle.base.backward import append_backward

paddle.enable_static()


class TestOptimizer(unittest.TestCase):
    def test_sgd_optimizer(self):
        def check_sgd_optimizer(optimizer_attr):
            init_program = framework.Program()
            program = framework.Program()
            block = program.global_block()
            mul_x = block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                lod_level=0,
                name="mul.x",
                optimize_attr=optimizer_attr,
            )
            mul_y = block.create_var(
                dtype="float32", shape=[10, 8], lod_level=0, name="mul.y"
            )
            mul_out = block.create_var(
                dtype="float32", shape=[5, 8], lod_level=0, name="mul.out"
            )
            mean_out = block.create_var(
                dtype="float32", shape=[1], lod_level=0, name="mean.out"
            )
            block.append_op(
                type="mul",
                inputs={"X": mul_x, "Y": mul_y},
                outputs={"Out": mul_out},
                attrs={"x_num_col_dims": 1},
            )
            block.append_op(
                type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out}
            )
            sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.01)
            opts, _ = sgd_optimizer.minimize(mean_out, init_program)
            return opts

        opts = check_sgd_optimizer({'learning_rate': 1.1})
        self.assertEqual(len(opts), 2)
        self.assertEqual([op.type for op in opts], ["scale", "sgd"])

        opts = check_sgd_optimizer({'learning_rate': 1.0})
        self.assertEqual(len(opts), 1)
        self.assertEqual([op.type for op in opts], ["sgd"])


class TestOptimizerBackwardApplygrad(unittest.TestCase):
    def test_sgd_optimizer(self):
        def check_sgd_optimizer(optimizer_attr):
            init_program = framework.Program()
            program = framework.Program()
            block = program.global_block()
            mul_x = block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                lod_level=0,
                name="mul.x",
                optimize_attr=optimizer_attr,
            )
            mul_y = block.create_var(
                dtype="float32", shape=[10, 8], lod_level=0, name="mul.y"
            )
            mul_out = block.create_var(
                dtype="float32", shape=[5, 8], lod_level=0, name="mul.out"
            )
            mean_out = block.create_var(
                dtype="float32", shape=[1], lod_level=0, name="mean.out"
            )
            block.append_op(
                type="mul",
                inputs={"X": mul_x, "Y": mul_y},
                outputs={"Out": mul_out},
                attrs={"x_num_col_dims": 1},
            )
            block.append_op(
                type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out}
            )
            sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.01)
            with framework.program_guard(program, init_program):
                p_g = sgd_optimizer.backward(mean_out)
                opts = sgd_optimizer.apply_gradients(p_g)
            return opts

        opts = check_sgd_optimizer({'learning_rate': 1.1})
        self.assertEqual(len(opts), 2)
        self.assertEqual([op.type for op in opts], ["scale", "sgd"])

        opts = check_sgd_optimizer({'learning_rate': 1.0})
        self.assertEqual(len(opts), 1)
        self.assertEqual([op.type for op in opts], ["sgd"])


class TestMomentumOptimizer(unittest.TestCase):
    class MockMomentum(paddle.optimizer.Momentum):
        def get_accumulators(self):
            return self._accumulators

        def get_velocity_str(self):
            return self._velocity_acc_str

    def test_vanilla_momentum_optimizer(self):
        init_program = framework.Program()
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32",
            shape=[5, 10],
            lod_level=0,
            name="mul.x",
            optimize_attr={'learning_rate': 1.1},
        )
        mul_y = block.create_var(
            dtype="float32", shape=[10, 8], lod_level=0, name="mul.y"
        )
        mul_out = block.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="mul.out"
        )
        block.append_op(
            type="mul",
            inputs={"X": mul_x, "Y": mul_y},
            outputs={"Out": mul_out},
            attrs={"x_num_col_dims": 1},
        )
        learning_rate = 0.01
        momentum_optimizer = self.MockMomentum(
            learning_rate=learning_rate, momentum=0.2
        )
        mean_out = block.create_var(
            dtype="float32", shape=[1], lod_level=0, name="mean.out"
        )
        block.append_op(
            type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out}
        )
        params_grads = append_backward(mean_out)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(momentum_optimizer.get_accumulators()), 0)
        with framework.program_guard(program, init_program):
            opts = momentum_optimizer.apply_gradients(params_grads)
        self.assertEqual(len(opts), 2)
        sgd_op = opts[-1]
        self.assertEqual([op.type for op in opts], ["scale", "momentum"])
        self.assertFalse(sgd_op.attr('use_nesterov'))

        # Check accumulators
        accumulators = momentum_optimizer.get_accumulators()
        self.assertEqual(len(accumulators), 1)
        self.assertTrue(momentum_optimizer.get_velocity_str() in accumulators)
        velocity_acc = accumulators[momentum_optimizer.get_velocity_str()]
        self.assertEqual(len(velocity_acc), 1)
        self.assertTrue(mul_x.name in velocity_acc)

        # Check init_program
        init_ops = init_program.global_block().ops
        self.assertEqual(len(init_ops), 2)
        self.assertEqual(init_ops[1].type, "fill_constant")
        self.assertAlmostEqual(init_ops[0].attr('value'), learning_rate)
        self.assertEqual(init_ops[0].type, "fill_constant")
        self.assertAlmostEqual(init_ops[1].attr('value'), 0.0)

    def test_nesterov_momentum_optimizer(self):
        init_program = framework.Program()
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32",
            shape=[5, 10],
            lod_level=0,
            name="mul.x",
            optimize_attr={'learning_rate': 1.1},
        )
        mul_y = block.create_var(
            dtype="float32", shape=[10, 8], lod_level=0, name="mul.y"
        )
        mul_out = block.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="mul.out"
        )
        block.append_op(
            type="mul",
            inputs={"X": mul_x, "Y": mul_y},
            outputs={"Out": mul_out},
            attrs={"x_num_col_dims": 1},
        )
        mean_out = block.create_var(
            dtype="float32", shape=[1], lod_level=0, name="mean.out"
        )
        block.append_op(
            type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out}
        )
        learning_rate = 0.01
        momentum_optimizer = self.MockMomentum(
            learning_rate=learning_rate, momentum=0.2, use_nesterov=True
        )
        params_grads = append_backward(mean_out)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(momentum_optimizer.get_accumulators()), 0)
        with framework.program_guard(program, init_program):
            opts = momentum_optimizer.apply_gradients(params_grads)
        self.assertEqual(len(opts), 2)
        sgd_op = opts[-1]
        self.assertEqual([op.type for op in opts], ["scale", "momentum"])
        self.assertTrue(sgd_op.attr('use_nesterov'))

        # Check accumulators
        accumulators = momentum_optimizer.get_accumulators()
        self.assertEqual(len(accumulators), 1)
        self.assertTrue(momentum_optimizer.get_velocity_str() in accumulators)
        velocity_acc = accumulators[momentum_optimizer.get_velocity_str()]
        self.assertEqual(len(velocity_acc), 1)
        self.assertTrue(mul_x.name in velocity_acc)

        # Check init_program
        init_ops = init_program.global_block().ops
        self.assertEqual(len(init_ops), 2)
        self.assertEqual(init_ops[1].type, "fill_constant")
        self.assertAlmostEqual(init_ops[0].attr('value'), learning_rate)
        self.assertEqual(init_ops[0].type, "fill_constant")
        self.assertAlmostEqual(init_ops[1].attr('value'), 0.0)


class TestAdamOptimizer(unittest.TestCase):
    class MockAdam(paddle.optimizer.Adam):
        def get_accumulators(self):
            return self._accumulators

        def get_moment1_str(self):
            return self._moment1_acc_str

        def get_moment2_str(self):
            return self._moment2_acc_str

    def test_adam_optimizer(self):
        init_program = framework.Program()
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32",
            shape=[5, 10],
            lod_level=0,
            name="mul.x",
            optimize_attr={'learning_rate': 1.1},
        )
        mul_y = block.create_var(
            dtype="float32", shape=[10, 8], lod_level=0, name="mul.y"
        )
        mul_out = block.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="mul.out"
        )
        block.append_op(
            type="mul",
            inputs={"X": mul_x, "Y": mul_y},
            outputs={"Out": mul_out},
            attrs={"x_num_col_dims": 1},
        )
        mean_out = block.create_var(
            dtype="float32", shape=[1], lod_level=0, name="mean.out"
        )
        block.append_op(
            type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out}
        )
        learning_rate = 0.01
        adam_optimizer = self.MockAdam(
            learning_rate=learning_rate, beta1=0.9, beta2=0.999
        )
        params_grads = append_backward(mean_out)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(adam_optimizer.get_accumulators()), 0)
        with framework.program_guard(program, init_program):
            opts = adam_optimizer.apply_gradients(params_grads)
        self.assertEqual(len(opts), 2)
        self.assertEqual([op.type for op in opts], ["scale", "adam"])

        # Check accumulators
        accumulators = adam_optimizer.get_accumulators()
        self.assertEqual(len(accumulators), 4)
        self.assertTrue(adam_optimizer.get_moment1_str() in accumulators)
        self.assertTrue(adam_optimizer.get_moment2_str() in accumulators)
        moment1_acc = accumulators[adam_optimizer.get_moment1_str()]
        moment2_acc = accumulators[adam_optimizer.get_moment2_str()]
        self.assertEqual(len(moment1_acc), 1)
        self.assertEqual(len(moment2_acc), 1)
        self.assertTrue(mul_x.name in moment1_acc)
        self.assertTrue(mul_x.name in moment2_acc)

        # Check init_program
        init_ops = init_program.global_block().ops
        self.assertEqual(len(init_ops), 5)
        self.assertEqual(init_ops[-1].type, "fill_constant")
        self.assertAlmostEqual(init_ops[0].attr('value'), learning_rate)


class TestGradientMergeOptimizer(unittest.TestCase):
    def net(self):
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32", shape=[5, 10], lod_level=0, name="mul.x"
        )
        mul_y = block.create_var(
            dtype="float32", shape=[10, 8], lod_level=0, name="mul.y"
        )
        mul_out = block.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="mul.out"
        )
        b1 = block.create_parameter(
            dtype="float32", shape=[5, 8], lod_level=0, name="b1"
        )
        b1_out = block.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="b1_out"
        )
        mean_out = block.create_var(
            dtype="float32", shape=[1], lod_level=0, name="mean.out"
        )
        block.append_op(
            type="mul",
            inputs={"X": mul_x, "Y": mul_y},
            outputs={"Out": mul_out},
            attrs={"x_num_col_dims": 1},
        )
        block.append_op(
            type="elementwise_add",
            inputs={"X": mul_out, "Y": b1},
            outputs={"Out": b1_out},
        )
        block.append_op(
            type="mean", inputs={"X": b1_out}, outputs={"Out": mean_out}
        )
        return mean_out

    def test_program_desc(
        self,
    ):
        cost = self.net()
        main_program = cost.block.program
        init_program = framework.Program()
        self.assertEqual(main_program.num_blocks, 1)
        self.assertEqual(len(cost.block.ops), 3)
        self.assertEqual(
            [op.type for op in cost.block.ops],
            ["mul", "elementwise_add", "mean"],
        )

        opt = paddle.optimizer.SGD(learning_rate=1.0)
        opt = paddle.incubate.optimizer.GradientMergeOptimizer(opt, k_steps=4)
        with framework.program_guard(main_program, init_program):
            ops, params_grads = opt.minimize(cost)

        self.assertEqual(main_program.num_blocks, 2)

        # main block
        self.assertEqual(len(cost.block.ops), 13)
        self.assertEqual(
            [op.type for op in cost.block.ops],
            [
                'mul',
                'elementwise_add',
                'mean',
                'fill_constant',
                'mean_grad',
                'elementwise_add_grad',
                'mul_grad',
                'increment',  # step += 1
                'elementwise_mod',  # step %= k_steps
                'equal',  # cond_var == (step == 0)
                'elementwise_add',
                'elementwise_add',
                'conditional_block',
            ],
        )

        # optimize block
        self.assertEqual(len(main_program.block(1).ops), 6)
        self.assertEqual(
            [op.type for op in main_program.block(1).ops],
            ['scale', 'scale', 'sgd', 'sgd', 'fill_constant', 'fill_constant'],
        )


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
