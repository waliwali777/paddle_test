# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from .pass_base import PassType, CPPPassWrapper, register_pass


@register_pass("fuse_elewise_add_act")
class FuseElementwiseAddActPass(CPPPassWrapper):
    def __init__(self):
        super(FuseElementwiseAddActPass, self).__init__()

    @property
    def cpp_name(self):
        return "fuse_elewise_add_act_pass"

    def _type(self):
        return PassType.FUSION_OPT


@register_pass("fuse_bn_act")
class FuseBatchNormActPass(CPPPassWrapper):
    def __init__(self):
        super(FuseBatchNormActPass, self).__init__()

    @property
    def cpp_name(self):
        return "fuse_bn_act_pass"

    def _type(self):
        return PassType.FUSION_OPT
