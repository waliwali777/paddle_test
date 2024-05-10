// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/operator_fusion/frontend/pattern_fuser.h"
#include "paddle/cinn/operator_fusion/frontend/pattern.h"

namespace cinn::fusion {

template <>
StmtPattern<FrontendStage> ConvertToStmtPattern(
    const PatternContent<FrontendStage>& content) {
  const auto& kind = GetOpPatternKind(content.op);
  if (kind == hlir::framework::kReduction) {
    return ReducePattern<FrontendStage>({content.op});
  } else if (kind == hlir::framework::kElementWise ||
             kind == hlir::framework::kBroadcast ||
             kind == hlir::framework::kInjective) {
    return TrivialPattern<FrontendStage>({content.op}, content.op);
  } else {
    return UnsupportPattern<FrontendStage>({content.op});
  }
}

template <>
StmtPattern<FrontendStage> MergePatternImpl(
    const ReduceTreePattern<FrontendStage>& first,
    const TrivialPattern<FrontendStage>& second) {
  return ReduceTreePlusTrivialPattern<FrontendStage>(first, second);
}

template <>
StmtPattern<FrontendStage> MergePatternImpl(
    const TrivialPattern<FrontendStage>& first,
    const ReducePattern<FrontendStage>& second) {
  const auto& contents =
      UniqueConcatVector(GetOpsInPattern<FrontendStage>(first),
                         GetOpsInPattern<FrontendStage>(second));
  return ReducePattern<FrontendStage>(contents);
}

template <>
StmtPattern<FrontendStage> MergePatternImpl(
    const TrivialPattern<FrontendStage>& first,
    const TrivialPattern<FrontendStage>& second) {
  const auto& contents =
      UniqueConcatVector(GetOpsInPattern<FrontendStage>(first),
                         GetOpsInPattern<FrontendStage>(second));
  return TrivialPattern<FrontendStage>(contents, second.sink_op());
}

template <>
StmtPattern<FrontendStage> MergePatternImpl(
    const TrivialPattern<FrontendStage>& first,
    const AnchorPattern<FrontendStage>& second) {
  return AnchorPattern<FrontendStage>(
      UniqueConcatVector(GetOpsInPattern<FrontendStage>(first),
                         GetOpsInPattern<FrontendStage>(second)),
      second.anchor(),
      second.anchor_state);
}

template <>
StmtPattern<FrontendStage> MergePatternImpl(
    const AnchorPattern<FrontendStage>& source,
    const AnchorPattern<FrontendStage>& dest) {
  const auto& contents =
      UniqueConcatVector(GetOpsInPattern<FrontendStage>(source),
                         GetOpsInPattern<FrontendStage>(dest));
  return AnchorPattern<FrontendStage>(
      contents, source.anchor(), AnchorState<FrontendStage>({}));
}

template <>
ExprPromise<FrontendStage> InitExprPromiseImpl(
    const TrivialPattern<FrontendStage>& pattern, pir::Value anchor) {
  return ExprPromise<FrontendStage>(anchor);
}

template <>
ExprPromise<FrontendStage> InitExprPromiseImpl(
    const ReducePattern<FrontendStage>& pattern, pir::Value anchor) {
  return ExprPromise<FrontendStage>(anchor);
}

template <>
TrivialPattern<FrontendStage> RecoverAnchorPatternToTrivial(
    const AnchorPattern<FrontendStage>& anchor_pattern) {
  PADDLE_ENFORCE_EQ(anchor_pattern.anchor_state.promise.size(),
                    1,
                    phi::errors::PreconditionNotMet(
                        "Can only recover AnchorPattern whose anchor_state "
                        "size is 1 (exact %d)",
                        anchor_pattern.anchor_state.promise.size()));

  return TrivialPattern<FrontendStage>(anchor_pattern.ops(),
                                       anchor_pattern.anchor().defining_op());
}

}  // namespace cinn::fusion
