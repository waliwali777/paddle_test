// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature SegmentPoolOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  // const auto& pooltype= paddle::any_cast<std::string>(ctx.Attr("pooltype"));
  // if(pooltype == "MEAN"){
  return KernelSignature(
      "segment_pool", {"X", "SegmentIds"}, {"pooltype"}, {"Out", "SummedIds"});
  // }else{
  //   return KernelSignature(
  //       "segment_pool", {"X", "SegmentIds"}, {"pooltype"}, {"Out"});
  // }
}

KernelSignature SegmentPoolGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "segment_pool_grad",
      {
          "X", "SegmentIds", "Out", "SummedIds", GradVarName("Out"),
      },
      {"pooltype"},
      {GradVarName("X")});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(segment_pool, phi::SegmentPoolOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(segment_pool_grad,
                           phi::SegmentPoolGradOpArgumentMapping);
