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
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core
from paddle.fluid.op import Operator


class TestLookupTableOp(OpTest):
    def setUp(self):
        self.op_type = "lookup_table"
        table = np.random.random((17, 31)).astype("float32")
        ids = np.random.randint(0, 17, 4).astype("int64")
        ids_expand = np.expand_dims(ids, axis=1)
        self.inputs = {'W': table, 'Ids': ids_expand}
        self.outputs = {'Out': table[ids]}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['W'], 'Out', no_grad_set=set('Ids'))


class TestLookupTableOpWithPadding(TestLookupTableOp):
    def test_check_output(self):
        ids = np.squeeze(self.inputs['Ids'])
        padding_idx = np.random.choice(ids, 1)[0]
        self.outputs['Out'][ids == padding_idx] = np.zeros(31)
        self.attrs = {'padding_idx': long(padding_idx)}
        self.check_output()

    def test_check_grad(self):
        # Since paddings are not trainable and fixed in forward, the gradient of 
        # paddings makes no sense and we don't test the gradient here.
        pass


class TestLookupTableIdsIsSelectedRows(OpTest):
    def check_with_place(self, place):
        scope = core.Scope()

        # create and initialize Grad Variable
        height = 10
        rows = [0, 4, 4, 7]
        row_numel = 12

        ids_selected_rows = scope.var('Ids').get_selected_rows()
        ids_selected_rows.set_height(height)
        ids_selected_rows.set_rows(rows)
        np_array = np.ones((len(rows), row_numel)).astype("float32")
        ids_tensor = ids_selected_rows.get_tensor()
        ids_tensor.set(np_array, place)

        # create and initialize W Variable
        W = scope.var('W').get_tensor()
        W_array = np.full((height, row_numel), 1.0).astype("float32")
        for i in range(height):
            W_array[i] *= i
        W.set(W_array, place)

        Out = scope.var('Out').get_selected_rows()
        Out_array = np.full((len(rows), row_numel), -1.0).astype("float32")
        Out.set_height(height)
        Out.set_rows(rows)
        Out_tensor = Out.get_tensor()
        Out_tensor.set(Out_array, place)

        # create and run concat_rows_op operator
        concat_rows_op = Operator("lookup_table", W='W', Ids='Ids', Out='Out')
        concat_rows_op.run(scope, place)

        # get and compare result
        result_array = np.array(Out_tensor)

        for idx, row in enumerate(rows):
            assert (row == result_array[idx]).all()

    def test_concat_rows(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_with_place(place)


if __name__ == "__main__":
    unittest.main()
