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

# TODO: Variables: make_channel
# TODO: Operators: send, close_channel, recv, go, select
from layers.control_flow import BlockGuard
from layer_helper import LayerHelper
import core
__all__ = [
    'Go',
    'make_channel',
    'channel_send',
    'channel_recv',
    'channel_close',
]


class Go(BlockGuard):
    def __init__(self, name=None):
        self.helper = LayerHelper("go", name=name)
        super(Go, self).__init__(self.helper.main_program)

    def __enter__(self):
        super(Go, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        self.construct_go_op()
        return super(Go, self).__exit__(exc_type, exc_val, exc_tb)

    def construct_go_op(self):
        main_program = self.helper.main_program
        go_block = main_program.current_block()
        parent_block = main_program.block(main_program.current_block()
                                          .parent_idx)

        x_name_list = set()
        out_vars = []
        for op in go_block.ops:
            # Iterate over all operators, get all the inputs
            # and add as input to the Go operator.
            for iname in op.input_names:
                for in_var_name in op.input(iname):
                    x_name_list.add(in_var_name)

            # Iterate over all operators , get all the outputs
            # add to the output list of Go operator only if
            # they exist in the parent block.
            for oname in op.output_names:
                for out_var_name in op.output(oname):
                    if out_var_name in parent_block.vars:
                        out_vars.add(parent_block.var(out_var_name))

        parent_block.append_op(
            type='go',
            inputs={'X': [parent_block.var(x_name) for x_name in x_name_list]},
            outputs={'Out': out_vars},
            attrs={'sub_block': go_block})


def make_channel(dtype, name, capacity=0):
    helper = LayerHelper('make_channel', **locals())
    main_program = helper.main_program
    make_channel_block = main_program.current_block()

    channel = helper.create_variable(type=core.VarDesc.VarType.CHANNEL)
    create_channel_op = make_channel_block.append_op(
        type="channel_create",
        outputs={"Channel": channel},
        attrs={"data_type": dtype,
               "capacity": capacity})

    return create_channel_op


def channel_send(channel, value):
    helper = LayerHelper('channel_send', **locals())
    main_program = helper.main_program
    channel_send_block = main_program.current_block()
    return_value = False

    channel_send_op = channel_send_block.append_op(
        type="channel_send", inputs={
            "Channel": channel,
            "Val": value,
        })

    return channel_send_op


def channel_recv(channel, dtype):
    helper = LayerHelper('channel_recv', **locals())
    main_program = helper.main_program
    channel_recv_block = main_program.current_block()
    return_value = helper.create_variable(type=dtype)

    channel_recv_op = channel_recv_block.append_op(
        type="channel_recv",
        inputs={"Channel": channel},
        outputs={"Output": return_value})

    return channel_recv_op


def channel_close(channel):
    helper = LayerHelper('channel_close', **locals())
    main_program = helper.main_program
    channel_close_block = main_program.current_block()

    return_value = False
    channel_close_op = channel_close_block.append_op(
        type="channel_close", inputs={"Channel": channel, })

    return channel_close_op
