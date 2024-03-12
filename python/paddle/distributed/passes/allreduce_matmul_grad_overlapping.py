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

import collections
import logging

from ..auto_parallel.static.dist_attribute import OperatorDistAttr
from ..auto_parallel.static.utils import (
    get_logger,
    naive_set_dist_op_attr_for_program_by_mesh,
)
from .pass_base import PassBase, register_pass
from .pass_utils import AutoParallelStreamType

logger = get_logger(logging.INFO)


# For allreduce pattern in the backward phase of column parallel linear:
#   dX, dY = matmul_grad(X, Y, dOut)
#   dX = c_allreduce_sum(dX)
# Split matmul_grad to 2 matmul:
#   dX = matmul(dOut, Y^T)
#   dX = c_allreduce_sum(dX)
#   dY = matmul(X^T, dOut)
#
# Then the c_allreduce_sum can overlap with the compute of dY.
@register_pass("allreduce_matmul_grad_overlapping")
class AllreduceMatmulGradOverlappingPass(PassBase):
    def __init__(self):
        super().__init__()
        self.op_namescope = "/auto_parallel/allreduce_matmul_grad_overlapping"
        self.set_attr("dist_context", None)

    def _check_self(self):
        if self.get_attr("dist_context") is None:
            return False
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        self.dist_context = self.get_attr("dist_context")
        block = main_program.global_block()

        matmul_grad_id_to_allreduce_id = (
            self._get_all_matmul_grad_and_allreduce_pairs(block)
        )
        logger.info(
            f"overlap matmul_grad and allreduce: {matmul_grad_id_to_allreduce_id}"
        )

        self._split_matmul_grad_and_multi_streaming_allreduce(
            block, matmul_grad_id_to_allreduce_id
        )

    def _get_all_matmul_grad_and_allreduce_pairs(self, block):
        ops = block.ops
        op_num = len(ops)
        matmul_grad_id_to_allreduce_id = collections.OrderedDict()
        for i, op_i in enumerate(ops):
            if (
                op_i.type == 'matmul_v2_grad'
                and op_i.attr("trans_x") is False
                and op_i.attr("trans_y") is False
            ):
                x_grad = op_i.output("X@GRAD")
                for j in range(i + 1, op_num):
                    op_j = ops[j]
                    if (
                        op_j.type == 'c_allreduce_sum'
                        and op_j.input("X") == x_grad
                    ):
                        matmul_grad_id_to_allreduce_id[i] = j
        return matmul_grad_id_to_allreduce_id

    def _insert_reshape_op(
        self, block, index, x, shape, op_role, chunk_id, var_out=None
    ):
        var_x = block.var(x)
        x_dist_attr = self.dist_context.get_tensor_dist_attr_for_program(var_x)

        if var_out is None:
            var_out = block.create_var(
                name=f"{x}@reshape.out",
                dtype=var_x.dtype,
                shape=shape,
                persistable=False,
            )
            ref_dist_attr = x_dist_attr
            ref_dist_attr.dims_mapping = x_dist_attr.dims_mapping[
                -len(shape) :
            ]  # unused
            self.dist_context.set_tensor_dist_attr_for_program(
                var_out, ref_dist_attr
            )

        out = var_out.name
        x_shape = f"{x}@reshape.xshape"
        var_x_shape = block.create_var(name=x_shape, dtype=var_x.dtype)
        self.dist_context.set_tensor_dist_attr_for_program(
            var_x_shape, x_dist_attr
        )

        reshape_op = block._insert_op_without_sync(
            index=index,
            type="reshape2",
            inputs={"X": x},
            outputs={"Out": out, "XShape": x_shape},
            attrs={
                "shape": shape,
                "op_role": op_role,
                'op_namescope': self.op_namescope,
            },
        )
        op_dist_attr = OperatorDistAttr(reshape_op.desc)
        op_dist_attr.inputs_dist_attrs = {
            x: x_dist_attr,
        }
        op_dist_attr.outputs_dist_attrs = {
            out: self.dist_context.get_tensor_dist_attr_for_program(var_out),
            x_shape: x_dist_attr,
        }
        naive_set_dist_op_attr_for_program_by_mesh(
            reshape_op,
            process_mesh=x_dist_attr.process_mesh,
            ctx=self.dist_context,
            chunk_id=chunk_id,
        )

        return out

    def _split_matmul_grad_and_multi_streaming_allreduce(
        self, block, matmul_grad_id_to_allreduce_id
    ):
        ops = block.ops

        for matmul_grad_id, allreduce_id in reversed(
            matmul_grad_id_to_allreduce_id.items()
        ):
            matmul_grad_op = ops[matmul_grad_id]
            allreduce_op = ops[allreduce_id]

            # NOTE(Sonder): Why move those operations to the back of matmul_v2?
            # When using amp_master_grad, the cast operation is inserted after matmul_grad.
            # However, when employing allreduce_matmul_grad_overlapping, the matmul_grad is
            # split into two matmul operations. In this case, some operations would access
            # uninitialized tensors. Therefore, we move the cast operation to the back of the
            # second matmul operation to avoid this problem.
            skip_overlapping = False
            moved_ops_idx = []
            moved_ops_output = []
            matmul_grad_output = matmul_grad_op.output('Y@GRAD')[0]

            for idx in range(matmul_grad_id + 1, allreduce_id):
                if matmul_grad_output in ops[idx].desc.input_arg_names():
                    moved_ops_idx.append(idx)
                    moved_ops_output.extend(ops[idx].desc.output_arg_names())
                else:
                    for input_name in ops[idx].desc.input_arg_names():
                        if input_name in moved_ops_output:
                            skip_overlapping = True

            if skip_overlapping:
                continue

            for i, idx in enumerate(moved_ops_idx):
                op = ops[idx]
                dist_attr = self.dist_context.get_op_dist_attr_for_program(op)

                op_inputs = op.desc.input_names()
                op_outputs = op.desc.output_names()

                op_inputs = {name: op.input(name) for name in op_inputs}
                op_outputs = {name: op.output(name) for name in op_outputs}

                op = block._insert_op_without_sync(
                    index=allreduce_id + 1 + i,
                    type=op.type,
                    inputs=op_inputs,
                    outputs=op_outputs,
                    attrs=op.all_attrs(),
                )

                self.dist_context.set_op_dist_attr_for_program(op, dist_attr)

            for i, idx in enumerate(moved_ops_idx):
                block._remove_op(idx - i, sync=False)
                allreduce_id -= 1

            tran_x = matmul_grad_op.attr("trans_x")
            assert (
                not tran_x
            ), f"matmul_grad(id={matmul_grad_id}) with tran_x == True is not supported for column parallel linear backward overlapping"
            tran_y = matmul_grad_op.attr("trans_y")
            assert (
                not tran_y
            ), f"matmul_grad(id={matmul_grad_id}) with tran_y == True is not supported for column parallel linear backward overlapping"

            allreduce_op.dist_attr.execution_stream = (
                AutoParallelStreamType.MP_STREAM.value
            )

            x = matmul_grad_op.input("X")[0]
            y = matmul_grad_op.input("Y")[0]
            out_grad = matmul_grad_op.input("Out@GRAD")[0]
            x_grad = matmul_grad_op.output("X@GRAD")[0]
            y_grad = matmul_grad_op.output("Y@GRAD")[0]
            op_role = matmul_grad_op.attr("op_role")

            # NOTE(Ruibiao): Required OP scheduling order: matmul(dOut, Y^T) -> c_allreduce_sum(dX) -> matmul(X^T, dOut).
            # c_allreduce_sum(dX) and matmul(X^T, dOut) cannot be swapped. Otherwise, after buffer_shared_inplace_pass
            # adding share_buffer OP before c_allreduce_sum, c_allreduce_sum will synchronous with comp-stream, and then
            # the matmul op before it cannot be overlapped.
            var_x = block.var(x)
            var_out_grad = block.var(out_grad)
            var_y_grad = block.var(y_grad)

            x_dims = var_x.shape
            out_grad_dims = var_out_grad.shape
            y_grad_dims = var_y_grad.shape

            assert len(x_dims) == len(
                out_grad_dims
            ), f"The rank of x must be equal to that of out_grad, but got x rank = {len(x_dims)} and out_grad rank = {len(out_grad_dims)}."
            if len(x_dims) > 2:
                assert (
                    x_dims[0:2] == out_grad_dims[0:2]
                ), f"The first two dimensions of x must be equal to that of out_grad, but got x_dims:{x_dims} and out_grad_dims:{out_grad_dims}."
                new_x_dims = [x_dims[0] * x_dims[1]] + list(x_dims[2:])
                new_out_grad_dims = [
                    out_grad_dims[0] * out_grad_dims[1]
                ] + list(out_grad_dims[2:])

            # NOTE(Ruibiao): Why insert reshape op here?
            # When the rank of input matrix is 3, MatmulGradKernel use reshape to fold the first two dimensions of x and out_grad (see FoldInitDims in matmul_grad_kernel_impl.h), and then calls blas.Matmul to calculate y_grad.
            # If we directly append matmul op to calculate y_grad without FoldInitDims, blas.BatchedGEMM is actually called in MatmulKernel, which has a larger cost than using blas.Matmul after dimension folding.
            # Therefore, we imitate MatmulGradKernel here by inserting reshape op before matmul.
            chunk_id = self.dist_context.get_op_dist_attr_for_program(
                matmul_grad_op
            ).chunk_id
            new_x = self._insert_reshape_op(
                block, allreduce_id + 1, x, new_x_dims, op_role, chunk_id
            )
            new_out_grad = self._insert_reshape_op(
                block,
                allreduce_id + 2,
                out_grad,
                new_out_grad_dims,
                op_role,
                chunk_id,
            )

            new_y_grad = f"{y_grad}@reshape.out"
            new_var_y_grad = block.create_var(
                name=new_y_grad,
                dtype=var_y_grad.dtype,
                persistable=False,
            )
            self.dist_context.set_tensor_dist_attr_for_program(
                new_var_y_grad,
                self.dist_context.get_tensor_dist_attr_for_program(var_y_grad),
            )

            matmul_op = block._insert_op_without_sync(
                index=allreduce_id + 3,
                type="matmul_v2",
                inputs={"X": new_x, "Y": new_out_grad},
                outputs={"Out": new_y_grad},
                attrs={
                    "trans_x": True,
                    "trans_y": False,
                    "op_role": op_role,
                    'op_namescope': self.op_namescope,
                },
            )

            ref_op_dist_attr = self.dist_context.get_op_dist_attr_for_program(
                matmul_grad_op
            )
            naive_set_dist_op_attr_for_program_by_mesh(
                matmul_op,
                ref_op_dist_attr.process_mesh,
                self.dist_context,
                chunk_id=ref_op_dist_attr.chunk_id,
            )

            self._insert_reshape_op(
                block,
                allreduce_id + 4,
                new_y_grad,
                y_grad_dims,
                op_role,
                chunk_id,
                var_y_grad,
            )

            matmul_op = block._insert_op_without_sync(
                index=matmul_grad_id + 1,
                type="matmul_v2",
                inputs={"X": out_grad, "Y": y},
                outputs={"Out": x_grad},
                attrs={
                    "trans_x": False,
                    "trans_y": True,
                    "op_role": op_role,
                    'op_namescope': self.op_namescope,
                },
            )
            naive_set_dist_op_attr_for_program_by_mesh(
                matmul_op,
                ref_op_dist_attr.process_mesh,
                self.dist_context,
                chunk_id=ref_op_dist_attr.chunk_id,
            )

            block._remove_op(matmul_grad_id, sync=False)
        block._sync_with_cpp()
