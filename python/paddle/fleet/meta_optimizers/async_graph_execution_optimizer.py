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

import paddle
from paddle.fluid import compiler
from .async_optimizer import AsyncOptimizer


class AsyncGraphExecutionOptimizer(AsyncOptimizer):
    def __init__(self, optimizer):
        super(AsyncGraphExecutionOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = []

    def _is_graph_out(self):
        return True

    def _try_to_compile(self, startup_program, main_program, loss):
        dist_strategy = self.get_distributed_strategy()

        build_strategy = dist_strategy.get_build_strategy()
        exec_strategy = dist_strategy.get_execute_strategy()

        self._compiled_program = compiler.CompiledProgram(main_program)

        self._compiled_program.with_data_parallel(
            loss_name=loss.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy,
            share_vars_from=None)

        return self._compiled_program

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        if startup_program == None:
            startup_program = paddle.default_startup_program()
        compiled_program = self._try_to_compile(startup_program,
                                                loss.block.program, loss)
        loss.block.program._graph = compiled_program

        # just return self.optimizer_ops and self.param_grads
        return None, None
