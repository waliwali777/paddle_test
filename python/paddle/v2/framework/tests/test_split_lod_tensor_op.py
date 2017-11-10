import unittest
import paddle.v2.framework.core as core
import numpy as np
import paddle.v2.framework.layers as layers
from paddle.v2.framework.framework import Program
from paddle.v2.framework.executor import Executor
from paddle.v2.framework.backward import append_backward_ops


class TestCPULoDTensorArrayOps(unittest.TestCase):
    def place(self):
        return core.CPUPlace()

    def test_lod_tensor_to_array_no_lod(self):
        tensor = core.LoDTensor()
        tensor.set(np.arange(10).reshape(10, 1).astype('int32'), self.place())

        mask_np = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0]).astype('bool')
        mask_np = np.expand_dims(mask_np, axis=1)

        mask = core.LoDTensor()
        mask.set(mask_np, self.place())

        expect_true_tensor = np.array([2, 3, 4, 5]).astype('int32')
        expect_true_tensor = np.expand_dims(expect_true_tensor, axis=1)
        expect_true = core.LoDTensor()
        expect_true.set(expect_true_tensor, self.place())

        expect_false_tensor = np.array([0, 1, 6, 7, 8, 9]).astype('int32')
        expect_false_tensor = np.expand_dims(expect_false_tensor, axis=1)

        expect_false = core.LoDTensor()
        expect_false.set(expect_false_tensor, self.place())

        self.main(
            tensor=tensor,
            mask=mask,
            expect_true=expect_true,
            expect_false=expect_false)

    def test_lod_tensor_to_array_level_0(self):
        tensor = core.LoDTensor()
        tensor.set(np.arange(10).reshape(10, 1).astype('int32'), self.place())
        tensor.set_lod([[0, 3, 9, 10]])

        mask_np = np.array([0, 1, 0]).astype('bool')
        mask_np = np.expand_dims(mask_np, axis=1)

        mask = core.LoDTensor()
        mask.set(mask_np, self.place())

        expect_true_tensor = np.array([3, 4, 5, 6, 7, 8]).astype('int32')
        expect_true_tensor = np.expand_dims(expect_true_tensor, axis=1)
        expect_true = core.LoDTensor()
        expect_true.set(expect_true_tensor, self.place())
        expect_true.set_lod([[0, 6]])

        expect_false_tensor = np.array([0, 1, 2, 9]).astype('int32')
        expect_false_tensor = np.expand_dims(expect_false_tensor, axis=1)
        expect_false_lod = [[0, 3, 4]]

        expect_false = core.LoDTensor()
        expect_false.set(expect_false_tensor, self.place())
        expect_false.set_lod(expect_false_lod)

        self.main(
            tensor=tensor,
            mask=mask,
            expect_true=expect_true,
            expect_false=expect_false)

    def main(self, tensor, mask, expect_true, expect_false, level=0):
        place = self.place()
        program = Program()
        x = layers.data(name='x', shape=[1], main_program=program)
        x.persistable = True

        y = layers.data(name='y', shape=[1], main_program=program)
        y.persistable = True

        out_true, out_false = layers.split_lod_tensor(
            input=x, mask=y, level=level, main_program=program)
        out_true.persistable = True
        out_false.persistable = True

        exe = Executor(place)
        scope = core.Scope()
        exe.run(program, feed={'x': tensor, 'y': mask}, scope=scope)

        var_true = scope.find_var(out_true.name).get_tensor()

        var_false = scope.find_var(out_false.name).get_tensor()

        self.check_tensor_same(var_true, expect_true)
        self.check_tensor_same(var_false, expect_false)

    def check_tensor_same(self, actual, expect):
        self.assertTrue(np.allclose(np.array(actual), np.array(expect)))
        self.assertEqual(actual.lod(), expect.lod())


if __name__ == '__main__':
    unittest.main()
