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
from paddle import base
from paddle.base.framework import Program, default_main_program, program_guard

paddle.enable_static()

main_program = default_main_program()


class TestProgram(unittest.TestCase):
    def test_program(self):
        b = main_program.current_block()
        self.assertEqual(-1, b.parent_idx)
        self.assertEqual(0, b.idx)

        b = main_program._create_block()
        self.assertEqual(1, b.idx)
        self.assertEqual(0, b.parent_idx)

        b = main_program._create_block()
        self.assertEqual(2, b.idx)
        self.assertEqual(1, b.parent_idx)

        main_program._rollback()

        b = main_program.current_block()
        self.assertEqual(1, b.idx)
        self.assertEqual(0, b.parent_idx)

        b = main_program._create_block()
        self.assertEqual(3, b.idx)
        self.assertEqual(1, b.parent_idx)

        main_program._rollback()
        b = main_program.current_block()
        self.assertEqual(1, b.idx)
        self.assertEqual(0, b.parent_idx)

    def test_program_clone(self):
        prog = Program()

        x = prog.global_block().create_var(
            name='X', shape=[1000, 784], dtype='float32'
        )

        y = prog.global_block().create_var(
            name='Y', shape=[784, 100], dtype='float32'
        )
        out = prog.global_block().create_var(name='Out', dtype='float32')
        prog.global_block().append_op(
            type="mul", inputs={'X': [x], 'Y': [y]}, outputs={'Out': [out]}
        )

        # FIXME(yuyang18): We manual compare the output string, since the order
        # of variable could be changed.
        print(prog)
        print(prog.clone())

    def test_parse_program_from_string(self):
        prog = Program()

        x = prog.global_block().create_var(
            name='X', shape=[1000, 784], dtype='float32'
        )

        y = prog.global_block().create_var(
            name='Y', shape=[784, 100], dtype='float32'
        )
        out = prog.global_block().create_var(name='Out', dtype='float32')
        prog.global_block().append_op(
            type="mul", inputs={'X': [x], 'Y': [y]}, outputs={'Out': [out]}
        )

        binary_str = prog.desc.serialize_to_string()
        prog_restored = Program.parse_from_string(binary_str)

        print(prog)
        print(prog_restored)

    def test_program_clone_with_parameter(self):
        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            d = paddle.static.data(name='x', shape=[-1, 784], dtype='float32')
            hidden = paddle.static.nn.fc(x=d, size=100)
            paddle.static.nn.fc(x=hidden, size=100)

        new_program = main_program.clone()
        self.assertNotEqual(0, len(new_program.blocks[0].all_parameters()))

    def test_program_all_parameters(self):
        program = base.default_main_program()
        data = paddle.static.data(name='x', shape=[None, 13], dtype='float32')
        hidden = paddle.static.nn.fc(x=data, size=10)
        loss = paddle.mean(hidden)
        paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

        # NOTE: here the parameters are fc_0.w_0 and fc_0.b_0
        param_list = program.all_parameters()
        self.assertEqual(len(param_list), 2)
        self.assertEqual(param_list[0].name, "fc_0.w_0")
        self.assertEqual(param_list[1].name, "fc_0.b_0")

    def test_prune_with_input_type_error(self):
        program = base.default_main_program()
        feed_var_names = [2, 3, 4]
        self.assertRaises(
            ValueError, program._prune_with_input, feed_var_names, []
        )

    def test_random_seed_error(self):
        program = base.default_main_program()
        with self.assertRaises(ValueError):
            paddle.seed("seed")

    def test_copy_info_from_error(self):
        program = base.default_main_program()
        self.assertRaises(TypeError, program._copy_param_info_from, "program")
        self.assertRaises(
            TypeError, program._copy_dist_param_info_from, "program"
        )


def build_program():
    main_program = paddle.static.Program()
    startuo_program = paddle.static.Program()
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(main_program, startuo_program):
            x = paddle.static.data(name='x', shape=[3, 2, 1])
            out = paddle.static.nn.fc(x=x, size=1, num_flatten_dims=2)
    return main_program


class TestProgramProto(unittest.TestCase):
    def test_update_op(self):
        program = build_program()
        a = program.desc.serialize_to_string()
        program.current_block().ops[0]._set_attr('use_mkldnn', True)
        self.assertTrue(program.desc.need_update())
        b = program.desc.serialize_to_string()
        self.assertFalse(a == b)

    def test_update_var(self):
        program = build_program()
        a = program.desc.serialize_to_string()
        program.current_block().var("x").desc.set_stop_gradient(False)
        self.assertTrue(program.desc.need_update())
        b = program.desc.serialize_to_string()
        self.assertFalse(a == b)

    def test_update_var_attr(self):
        program = build_program()
        a = program.desc.serialize_to_string()
        program.current_block().var("x").desc._set_attr("a", 1)
        self.assertTrue(program.desc.need_update())
        b = program.desc.serialize_to_string()
        self.assertFalse(a == b)


class TestProgramHash(unittest.TestCase):
    def build_program(self):
        main_program = paddle.static.Program()
        startuo_program = paddle.static.Program()
        with paddle.utils.unique_name.guard():
            with paddle.static.program_guard(main_program, startuo_program):
                x = paddle.static.data(name='x', shape=[3, 2, 1])
                out = paddle.static.nn.fc(x=x, size=1, num_flatten_dims=2)
        return main_program

    def test_program_need_update(self):
        program = self.build_program()
        self.assertTrue(program.desc.need_update())
        program.desc.flush()
        self.assertFalse(program.desc.need_update())

    def test_program_hash_equal(self):
        programs = []
        for i in range(2):
            programs.append(self.build_program())
        program1, program2 = programs[0], programs[1]
        # why not write as below?
        # since the callstack attribute are not equal
        # program1 = self.build_program()
        # program2 = self.build_program()

        self.assertTrue(program1.desc.need_update())
        self.assertTrue(program2.desc.need_update())
        # two program with same content
        self.assertFalse(id(program1) == id(program2))
        # print(program1, program2)
        self.assertTrue(
            program1.desc.cached_hash_str() == program2.desc.cached_hash_str()
        )

        self.assertFalse(program1.desc.need_update())
        self.assertFalse(program2.desc.need_update())

    def test_program_clone(self):
        program = self.build_program()
        program_clone = program.clone()

        self.assertFalse(id(program) == id(program_clone))
        self.assertTrue(
            program.desc.cached_hash_str()
            == program_clone.desc.cached_hash_str()
        )

    def test_program_update(self):
        program = self.build_program()
        hash1 = program.desc.cached_hash_str()
        id1 = id(program)
        # change mul's attr
        program.current_block().ops[0]._set_attr('use_mkldnn', True)
        program.current_block().ops[0]._set_attr('scale_x', 2.0)
        hash2 = program.desc.cached_hash_str()
        id2 = id(program)
        self.assertTrue(id1 == id2)
        self.assertFalse(hash1 == hash2)


if __name__ == '__main__':
    unittest.main()
