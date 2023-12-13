//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>

#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace details {
class OpHandleBase;

// Wraps ir::Node and provide helper utilities.
// It's responsible for populating necessary fields of ir::Node.
//
// VarHandleBase is the var node in the dependency graph.
// A variable can only be generated by a single operator. i.e.
// This is a single assignment graph.
struct VarHandleBase {
  // Owned by `node`. No need to be deleted explicitly.
  explicit VarHandleBase(ir::Node* node) : node_(node) {
    node_->WrappedBy(this);
  }

  TEST_API virtual ~VarHandleBase();

  virtual std::string DebugString() const = 0;
  virtual const std::string& Name() const = 0;

  void AddInput(OpHandleBase* in, ir::Node* node) {
    node_->inputs.clear();
    node_->inputs.push_back(node);
    generated_op_ = in;
  }

  void AddOutput(OpHandleBase* out, ir::Node* node) {
    if (pending_ops_.find(out) == pending_ops_.end()) {
      PADDLE_ENFORCE_NOT_NULL(out,
                              platform::errors::InvalidArgument(
                                  "The output added to VarHandle %s is NULL.",
                                  this->Node()->Name()));
      pending_ops_.insert(out);
      node_->outputs.push_back(node);
    }
  }

  void RemoveOutput(OpHandleBase* out, ir::Node* node) {
    pending_ops_.erase(out);
    node_->outputs.erase(
        std::remove(node_->outputs.begin(), node_->outputs.end(), node),
        node_->outputs.end());
  }

  void ClearGeneratedOp() {
    generated_op_ = nullptr;
    node_->inputs.clear();
  }

  OpHandleBase* GeneratedOp() { return generated_op_; }

  const OpHandleBase* GeneratedOp() const { return generated_op_; }

  const std::unordered_set<OpHandleBase*>& PendingOps() const {
    return pending_ops_;
  }

  ir::Node* Node() { return node_; }

  const ir::Node* Node() const { return node_; }

 protected:
  // The operator who generate this variable. nullptr if the variable
  // is a root node.
  OpHandleBase* generated_op_{nullptr};

  // Operators which depend on this variable ready.
  std::unordered_set<OpHandleBase*> pending_ops_;
  ir::Node* node_;
};

// VarHandle is actually a single version of Runtime Variable.
// Variable in Runtime mapped to many VarHandles in Graph.
// Each assignment will generate a new var handle with newer version.
//
// NOTE: runtime variables have place.
struct VarHandle : public VarHandleBase {
  DISABLE_COPY_AND_ASSIGN(VarHandle);

 public:
  TEST_API virtual ~VarHandle();

  TEST_API std::string DebugString() const override;

  VarHandle(ir::Node* node,
            size_t version,
            size_t scope_index,
            std::string name,
            platform::Place place)
      : VarHandleBase(node),
        version_(version),
        scope_idx_(scope_index),
        name_(std::move(name)),
        place_(std::move(place)) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  bool HasEvent() { return has_event_; }

  const gpuEvent_t& GetEvent() {
    PADDLE_ENFORCE_EQ(
        HasEvent(),
        true,
        platform::errors::PreconditionNotMet(
            "The cuda event is not set, maybe InitCUDA() is not called."));
    return event_;
  }

  void SetGenerateEvent(const gpuEvent_t& event) {
    has_event_ = true;
    event_ = event;
  }
#endif

  // version field currently is not used, however, just store the version to
  // debug easily.
 private:
  size_t version_;
  size_t scope_idx_;
  std::string name_;
  platform::Place place_;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // Only when this event is triggered, var is generated.
  gpuEvent_t event_;
  bool has_event_{false};
#endif

 public:
  bool IsTheSameVar(const VarHandle& o) const {
    return o.generated_op_ == generated_op_ && o.name_ == name_ &&
           o.scope_idx_ == scope_idx_;
  }

  size_t version() const { return version_; }
  size_t scope_idx() const { return scope_idx_; }
  const std::string& Name() const override { return name_; }
  const std::string& name() const { return name_; }
  const platform::Place& place() const { return place_; }
};

// Dummy Variable. It is used to represent dependencies between operators
struct DummyVarHandle : public VarHandleBase {
  explicit DummyVarHandle(ir::Node* node) : VarHandleBase(node) {}

  TEST_API virtual ~DummyVarHandle();

  TEST_API std::string DebugString() const override;

 public:
  const std::string& Name() const override { return name_; }
  std::string name_{"DummyVar"};
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
