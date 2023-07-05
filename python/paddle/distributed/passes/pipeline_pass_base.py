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

from paddle.fluid import core

from .pass_base import PassBase
from .pass_utils import get_skip_gc_vars


class PipelinePassBase(PassBase):
    def __init__(self):
        super().__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def create_job_list(self):
        """
        An interface that MUST be implemented by subclasses.
        """
        pass

    def partial_programs(self, program):
        """
        An interface that MUST be implemented by subclasses.
        The return value MUST be two lists, one is a list of names(str), another
        is a list of values.
        For example:
        return ["lr", "forward", "backward", "optimizer"], [lr_prog, fwd_prog, bwd_prog, opt_prog]
        or
        return ["forward"], [fwd_prog]
        """
        pass

    def _apply_single_impl(self, main_program, startup_program, context):
        """
        The shared process is implemented in this function and new subclass only need
        to implement two interfaces above, 'create_job_list' and 'partial_programs'.
        """
        type_name, type_values = self.partial_programs(main_program)

        job_list = self.create_job_list()

        # Following is a shared gc process for base class.
        type_gc_vars = get_skip_gc_vars(type_values)
        name_to_gc_vars = {}
        for name, gc_var in zip(type_name, type_gc_vars):
            name_to_gc_vars[name] = gc_var

        for job in job_list:
            job.set_skip_gc_vars(name_to_gc_vars[job.type()])

        type_to_program = {}
        for name, value in zip(type_name, type_values):
            type_to_program[name] = value.desc

        plan = core.Plan(job_list, type_to_program)
        context.set_attr("plan", plan)
