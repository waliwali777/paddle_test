/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License. */

#pragma once

#include <future>
#include <map>
#include <vector>

#include "paddle/pten/core/context.h"

// TODO(wilber): need to replace fluid place.
#include "paddle/fluid/platform/place.h"

namespace pten {

using Place = paddle::platform::Place;

template <typename Place>
struct DefaultDeviceContextType;

template <>
struct DefaultDeviceContextType<paddle::platform::CPUPlace> {
  using TYPE = CPUContext;
};

/*! \brief device context pool singleton */
class DeviceContextPool {
 public:
  explicit DeviceContextPool(const std::vector<Place>& places);

  static DeviceContextPool& Instance() {
    PADDLE_ENFORCE_NOT_NULL(pool,
                            paddle::platform::errors::PreconditionNotMet(
                                "Need to Create DeviceContextPool firstly!"));
    return *pool;
  }

  /*! \brief  Create should only called by Init function */
  static DeviceContextPool& Init(const std::vector<Place>& places) {
    if (pool == nullptr) {
      pool = new DeviceContextPool(places);
    }
    return *pool;
  }

  static void SetPool(DeviceContextPool* dev_pool) { pool = dev_pool; }

  /*! \brief  Return handle of single device context. */
  DeviceContext* Get(const Place& place);

  template <typename Place>
  const typename DefaultDeviceContextType<Place>::TYPE* GetByPlace(
      const Place& place) {
    return reinterpret_cast<
        const typename DefaultDeviceContextType<Place>::TYPE*>(Get(place));
  }

  size_t size() const { return device_contexts_.size(); }

 private:
  static DeviceContextPool* pool;
  std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>
      device_contexts_;
  DISABLE_COPY_AND_ASSIGN(DeviceContextPool);
};

}  // namespace pten
