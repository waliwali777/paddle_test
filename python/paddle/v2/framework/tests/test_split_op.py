import unittest
import numpy as np
from op_test import OpTest


class TestSplitOp(OpTest):
    def setUp(self):
        self.op_type = "split"
        axis = 0
        indices = 2
        sections = [1, 3]
        x = np.random.random((4, 2)).astype('float32')
        out = np.split(x, indices, axis)
        self.inputs = {'X': x}
        self.attrs = {'axis': axis, 'num': indices}
        self.outputs = {'Out': [('out%d' % i, out[i]) \
            for i in xrange(len(out))]}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], ['out0', 'out1'])


if __name__ == '__main__':
    unittest.main()
