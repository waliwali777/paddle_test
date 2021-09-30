// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/var_desc.h"

#include "cinn/frontend/paddle/cpp/block_desc.h"
#include "cinn/frontend/paddle/cpp/desc_api.h"
#include "cinn/frontend/paddle/cpp/op_desc.h"
#include "cinn/frontend/paddle/cpp/program_desc.h"
#include "cinn/frontend/paddle/cpp/var_desc.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

// Why use framework::VarDesc* rather than const framework::VarDesc& here?
// framework::VarDesc lack of many API like clear(), etc. On the other hand,
// the paddle node return framework::Desc* even if the node is const
void TransformVarDescToCinn(framework::VarDesc* pb_desc,
                            ::cinn::frontend::paddle::cppVarDesc* cpp_desc);

void TransformVarDescFromCinn(
    const ::cinn::frontend::paddle::cppVarDesc& cpp_desc,
    framework::VarDesc* pb_desc);

void TransformOpDescToCinn(framework::OpDesc* pb_desc,
                           ::cinn::frontend::paddle::cppOpDesc* cpp_desc);

void TransformOpDescFromCinn(
    const ::cinn::frontend::paddle::cppOpDesc& cpp_desc,
    framework::OpDesc* pb_desc);

void TransformBlockDescToCinn(framework::BlockDesc* pb_desc,
                              ::cinn::frontend::paddle::cppBlockDesc* cpp_desc);

void TransformBlockDescFromCinn(
    const ::cinn::frontend::paddle::cppBlockDesc& cpp_desc,
    framework::BlockDesc* pb_desc);

void TransformProgramDescToCinn(
    framework::ProgramDesc* pb_desc,
    ::cinn::frontend::paddle::cppProgramDesc* cpp_desc);

void TransformProgramDescFromCinn(
    const ::cinn::frontend::paddle::cppProgramDesc& cpp_desc,
    framework::ProgramDesc* pb_desc);

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle
