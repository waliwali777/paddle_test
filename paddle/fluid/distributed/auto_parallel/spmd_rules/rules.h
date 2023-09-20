// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/common.h"
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/cross_entropy_with_softmax_spmd_rule.h"
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/embedding_spmd_rule.h"
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/layer_norm_spmd_rule.h"
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/reduction_spmd_rule.h"
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/replicated_spmd_rule.h"
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/reshape_spmd_rule.h"
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/softmax_spmd_rule.h"
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/split_spmd_rule.h"
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/transpose_spmd_rule.h"

// TODO(ljz) Automatic this process in cmake file.
namespace paddle {
namespace distributed {
namespace auto_parallel {

// reduction rules
REGISTER_SPMD_RULE(all, ReductionSPMDRule);
REGISTER_SPMD_RULE(amax, ReductionSPMDRule);
REGISTER_SPMD_RULE(amin, ReductionSPMDRule);
REGISTER_SPMD_RULE(any, ReductionSPMDRule);
REGISTER_SPMD_RULE(frobenius_norm, ReductionSPMDRule);
REGISTER_SPMD_RULE(max, ReductionSPMDRule);
REGISTER_SPMD_RULE(mean, ReductionSPMDRule);
REGISTER_SPMD_RULE(min, ReductionSPMDRule);
REGISTER_SPMD_RULE(prod, ReductionSPMDRule);
REGISTER_SPMD_RULE(sum, ReductionSPMDRule);

// layer_norm rule
REGISTER_SPMD_RULE(layer_norm, LayerNormSPMDRule);

// replicated rule
REGISTER_SPMD_RULE(replicated, ReplicatedSPMDRule);

// embedding rule
REGISTER_SPMD_RULE(embedding, EmbeddingSPMDRule);
REGISTER_SPMD_RULE(lookup_table_v2, EmbeddingSPMDRule);

// softmax rule
REGISTER_SPMD_RULE(softmax, SoftmaxSPMDRule);
REGISTER_SPMD_RULE(log_softmax, SoftmaxSPMDRule);

// cross_entropy_with_softmax
REGISTER_SPMD_RULE(cross_entropy_with_softmax, CrossEntropyWithSoftmaxSPMDRule);
REGISTER_SPMD_RULE(softmax_with_cross_entropy, CrossEntropyWithSoftmaxSPMDRule);

// split rule
REGISTER_SPMD_RULE(split, SplitSPMDRule);
REGISTER_SPMD_RULE(split_with_num, SplitSPMDRule);

// transpose rule
REGISTER_SPMD_RULE(transpose, TransposeSPMDRule);

// reshape rule
REGISTER_SPMD_RULE(reshape, ReshapeSPMDRule);

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
