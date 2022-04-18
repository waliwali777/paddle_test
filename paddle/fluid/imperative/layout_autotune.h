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

#pragma once
#include <glog/logging.h>
#include <memory>
#include <set>
#include "paddle/fluid/framework/op_info.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/compat/type_defs.h"

namespace phi {
namespace autotune {

class DenseTensor;

using DataLayout = paddle::experimental::DataLayout;

class LayoutAutoTune {
 public:
  static LayoutAutoTune& Instance() {
    static LayoutAutoTune layout_autoTune;
    return layout_autoTune;
  }

  bool UseLayoutAutoTune() {
#if defined(PADDLE_WITH_CUDA)
    if (!phi::backends::gpu::TensorCoreAvailable()) {
      LOG(INFO) << "Layout AutoTuning is not available.";
      return false;
    } else {
      return use_layout_autotune_;
    }
#else
    return false;
#endif
  }

  void EnableLayoutAutoTune() { use_layout_autotune_ = true; }

  void DisableLayoutAutoTune() { use_layout_autotune_ = false; }

  bool IsLightlyLayoutSensitive(const std::string& op_type) {
    return lightly_layout_sensitive_ops_.count(op_type) != 0;
  }

  bool IsLayoutAgnostic(const std::string& op_type) {
    return layout_agnostic_ops_.count(op_type) != 0;
  }

  DataLayout GetDesiredLayout() { return layout_; }

  void SetDesiredLayout(const DataLayout& layout) { layout_ = layout; }

 private:
  LayoutAutoTune() {
    const auto& op_info = paddle::framework::OpInfoMap::Instance().map();
    for (auto it = op_info.begin(); it != op_info.end(); it++) {
      // only record forwrd operators
      if (it->first.find("_grad") != std::string::npos) {
        continue;
      }

      // some normalization operators such as instance_norm and layer_norm
      // do not have data_format attr, but are layout sensitive.
      if (it->first.find("norm") != std::string::npos) {
        layout_agnostic_ops_.emplace(it->first);
        continue;
      }

      auto* attr_checker = it->second.Checker();
      if (attr_checker) {
        auto attrs = attr_checker->GetDefaultAttrMap();
        if (attrs.find("data_format") != attrs.end() ||
            attrs.find("data_layout") != attrs.end()) {
          VLOG(4) << "Heavily layout sensitive OP: " << it->first;
          heavily_layout_sensitive_ops_.emplace(it->first);
          continue;
        }

        // Attribute name is fuzzy matched, such as start and start_axis.
        bool layout_agnostic = true;
        for (auto& attr : attrs) {
          auto attr_name = attr.first;
          VLOG(6) << "OP: " << it->first << " Attr Name: " << attr_name;
          if (attr_name.find("axis") != std::string::npos ||
              attr_name.find("axes") != std::string::npos ||
              attr_name.find("dim") != std::string::npos ||
              attr_name.find("start") != std::string::npos ||
              attr_name.find("end") != std::string::npos) {
            VLOG(4) << "Lightly layout sensitive OP: " << it->first;
            layout_agnostic = false;
            lightly_layout_sensitive_ops_.emplace(it->first);
            break;
          }
        }

        if (layout_agnostic) {
          VLOG(4) << "Layout agnostic_ops: " << it->first;
          layout_agnostic_ops_.emplace(it->first);
        }
      }
    }

    VLOG(3) << "The number of layout agnostic OPs: "
            << layout_agnostic_ops_.size() << ", heavily layout sensitive OPs: "
            << heavily_layout_sensitive_ops_.size()
            << ", lightly layout sensitive OPs: "
            << lightly_layout_sensitive_ops_.size();
  }

  bool use_layout_autotune_{false};

  std::unordered_set<std::string> layout_agnostic_ops_{};

  std::unordered_set<std::string> heavily_layout_sensitive_ops_{};

  std::unordered_set<std::string> lightly_layout_sensitive_ops_{};

  DataLayout layout_{DataLayout::UNDEFINED};
};

}  // namespace autotune
}  // namespace phi
