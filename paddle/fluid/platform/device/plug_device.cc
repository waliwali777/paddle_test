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

#include "paddle/fluid/platform/device/device_base.h"
#include "paddle/fluid/platform/device/device_manager.h"
#include "paddle/fluid/platform/device/event.h"
#include "paddle/fluid/platform/device/stream.h"
#include "paddle/fluid/platform/device_context.h"

static bool operator==(const C_Device_st& d1, const C_Device_st& d2) {
  return d1.id == d2.id;
}

namespace paddle {
namespace platform {

class PluggableDevice : public DeviceInterface {
 public:
  PluggableDevice(const std::string& type, int priority, bool is_pluggable,
                  std::unique_ptr<C_DeviceInterface> pimpl, void* dso_handle)
      : DeviceInterface(type, priority, is_pluggable),
        pimpl_(std::move(pimpl)),
        dso_handle_(dso_handle) {
    // TODO(wangran16): avoid call initialize on constructor
    Initialize();
  }

  ~PluggableDevice() override {
    // TODO(wangran16): avoid call finalize on deconstructor
    Finalize();
  }

  size_t VisibleDevicesCount() override {
    size_t count;
    if (pimpl_->visible_devices_count(&count) != C_SUCCESS) {
      count = 0;
    }
    return count;
  }

  std::vector<size_t> ListVisibleDevices() override {
    std::vector<size_t> devices(VisibleDevicesCount());
    pimpl_->visible_devices(devices.size());
    return devices;
  }

  C_DeviceInterface* Impl() { return pimpl_.get(); }

  void SynchronizeDevice(size_t dev_id) override {
    PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(pimpl_->synchronize_device(dev_id));
  }

  void Initialize() override {
    if (pimpl_->initialize && pimpl_->initialize() != C_SUCCESS) {
      LOG(ERROR) << "Initialize " + Type() + " Failed\n";
      exit(-1);
    }

    std::vector<size_t> visible_devices = ListVisibleDevices();
    for (auto dev_id : visible_devices) {
      InitDevice(dev_id);
    }
  }

  void Finalize() override {
    std::vector<size_t> visible_devices = ListVisibleDevices();
    for (auto dev_id : visible_devices) {
      // SetDevice(dev_id);
      // SynchronizeDevice(dev_id);
      DeInitDevice(dev_id);
    }

    bool ok = true;
    if (pimpl_->finalize && pimpl_->finalize() != C_SUCCESS) {
      LOG(ERROR) << "Finalize " + Type() + " Failed\n";
      ok = false;
    }
    if (dso_handle_) {
      dlclose(dso_handle_);
      dso_handle_ = nullptr;
    }
    if (!ok) {
      exit(1);
    }
  }

  void InitDevice(size_t dev_id) override {
    if (pimpl_->init_device) {
      // Core set logical id, and Plugin replace it with physical id
      PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(pimpl_->init_device(dev_id));
    }
  }

  void DeInitDevice(size_t dev_id) override {
    if (pimpl_->deinit_device) {
      PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(pimpl_->deinit_device(dev_id));
    }
  }

  void SetDevice(size_t dev_id) override {
    PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(pimpl_->set_device(dev_id));
  }

  int GetDevice() override {
    C_Device_st device;
    PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(pimpl_->get_device(&device));
    return device.id;
  }

  void CreateStream(size_t dev_id, stream::Stream* stream,
                    const stream::Stream::Priority& priority =
                        stream::Stream::Priority::kNormal,
                    const stream::Stream::Flag& flag =
                        stream::Stream::Flag::kDefaultFlag) override {
    if (priority != stream::Stream::Priority::kNormal ||
        flag != stream::Stream::Flag::kDefaultFlag) {
      PADDLE_THROW(platform::errors::Unavailable(
          "priority != stream::Stream::Priority::kNormal || flag != "
          "stream::Stream::Flag::kDefaultFlag is not allowed on "
          "PluggableDevice."));
    }
    C_Stream c_stream;
    PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(
        pimpl_->create_stream(dev_id, &c_stream));
    stream->set_stream(c_stream);
  }

  void DestroyStream(size_t dev_id, stream::Stream* stream) override {
    PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(pimpl_->destroy_stream(
        dev_id, reinterpret_cast<C_Stream>(stream->raw_stream())));
  }

  void SynchronizeStream(size_t dev_id, const stream::Stream* stream) override {
    PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(pimpl_->synchronize_stream(
        dev_id, reinterpret_cast<C_Stream>(stream->raw_stream())));
  }

  bool QueryStream(size_t dev_id, const stream::Stream* stream) override {
    if (!pimpl_->query_stream) {
      SynchronizeStream(dev_id, stream);
      return true;
    }
    if (pimpl_->query_stream(dev_id, reinterpret_cast<C_Stream>(
                                         stream->raw_stream())) == C_SUCCESS) {
      return true;
    }
    return false;
  }

  void AddCallback(size_t dev_id, stream::Stream* stream,
                   stream::Stream::Callback* callback) override {
    if (!pimpl_->stream_add_callback) {
      PADDLE_THROW(platform::errors::Unavailable(
          "AddCallback is not supported on " + Type() + "."));
    } else {
      PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(pimpl_->stream_add_callback(
          dev_id, reinterpret_cast<C_Stream>(stream->raw_stream()),
          [](C_Device device, C_Stream stream, void* user_data,
             C_Status* status) {
            std::unique_ptr<std::function<void()>> func(
                reinterpret_cast<std::function<void()>*>(user_data));
            (*func)();
          },
          callback));
    }
  }

  void CreateEvent(size_t dev_id, event::Event* event,
                   event::Event::Flag flags) override {
    C_Event c_event;

    PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(
        pimpl_->create_event(dev_id, &c_event));
    event->set_event(c_event);
  }

  void DestroyEvent(size_t dev_id, event::Event* event) override {
    PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(pimpl_->destroy_event(
        dev_id, reinterpret_cast<C_Event>(event->raw_event())));
  }

  void RecordEvent(size_t dev_id, const event::Event* event,
                   const stream::Stream* stream) override {
    PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(pimpl_->record_event(
        dev_id, reinterpret_cast<C_Stream>(stream->raw_stream()),
        reinterpret_cast<C_Event>(event->raw_event())));
  }

  void SynchronizeEvent(size_t dev_id, const event::Event* event) override {
    PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(pimpl_->synchronize_event(
        dev_id, reinterpret_cast<C_Event>(event->raw_event())));
  }

  bool QueryEvent(size_t dev_id, const event::Event* event) override {
    if (!pimpl_->query_event) {
      SynchronizeEvent(dev_id, event);
      return true;
    }
    if (pimpl_->query_event(dev_id, reinterpret_cast<C_Event>(
                                        event->raw_event())) == C_SUCCESS) {
      return true;
    }
    return false;
  }

  void StreamWaitEvent(size_t dev_id, const stream::Stream* stream,
                       const event::Event* event) override {
    PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(pimpl_->stream_wait_event(
        dev_id, reinterpret_cast<C_Stream>(stream->raw_stream()),
        reinterpret_cast<C_Event>(event->raw_event())));
  }

  void MemoryCopy(size_t dev_id, void* dst, const void* src, size_t size,
                  MemoryCpyKind kind,
                  const stream::Stream* stream = nullptr) override {
    auto place = platform::PluggableDevicePlace(Type(), dev_id);

    if (kind == MemoryCpyKind::HostToDevice) {
      if (stream && stream->raw_stream() && pimpl_->async_memory_copy_h2d) {
        C_Stream c_stream = reinterpret_cast<C_Stream>(stream->raw_stream());
        PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(
            pimpl_->async_memory_copy_h2d(dev_id, c_stream, dst, src, size));
      } else {
        platform::DeviceContextPool& pool =
            platform::DeviceContextPool::Instance();
        pool.Get(place)->Wait();
        PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(
            pimpl_->memory_copy_h2d(dev_id, dst, src, size));
      }
    } else if (kind == MemoryCpyKind::DeviceToHost) {
      if (stream && stream->raw_stream() && pimpl_->async_memory_copy_d2h) {
        C_Stream c_stream = reinterpret_cast<C_Stream>(stream->raw_stream());
        PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(
            pimpl_->async_memory_copy_d2h(dev_id, c_stream, dst, src, size));
      } else {
        platform::DeviceContextPool& pool =
            platform::DeviceContextPool::Instance();
        pool.Get(place)->Wait();
        PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(
            pimpl_->memory_copy_d2h(dev_id, dst, src, size));
      }
    } else if (kind == MemoryCpyKind::DeviceToDevice) {
      if (stream && stream->raw_stream() && pimpl_->async_memory_copy_d2d) {
        C_Stream c_stream = reinterpret_cast<C_Stream>(stream->raw_stream());
        PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(
            pimpl_->async_memory_copy_d2d(dev_id, c_stream, dst, src, size));
      } else {
        platform::DeviceContextPool& pool =
            platform::DeviceContextPool::Instance();
        pool.Get(place)->Wait();
        PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(
            pimpl_->memory_copy_d2d(device, dst, src, size));
      }
    } else {
      PADDLE_THROW(platform::errors::Unavailable("Unknow MemoryCpyKind."));
    }
  }

  void MemoryCopyPeer(const Place& dst_place, void* dst, size_t src_dev_id,
                      const void* src, size_t size,
                      const stream::Stream* stream = nullptr) override {
    int dst_dev_id = PlaceToId(dst_place);

    if (stream && stream->raw_stream()) {
      if (!pimpl_->async_memory_copy_p2p) {
        MemoryCopyPeer(dst_place, dst, src_dev_id, src, size);
      } else {
        PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(pimpl_->async_memory_copy_p2p(
            dst_dev_id, src_device,
            reinterpret_cast<C_Stream>(stream->raw_stream()), dst, src, size));
      }
    } else {
      if (!pimpl_->memory_copy_p2p) {
        std::unique_ptr<void> p(new uint8_t[size]);
        MemoryCopy(src_dev_id, p.get(), src, size, MemoryCpyKind::DeviceToHost);
        MemoryCopy(dst_dev_id, dst, p.get(), size, MemoryCpyKind::HostToDevice);
      } else {
        auto src_place = platform::PluggableDevicePlace(Type(), src_dev_id);
        platform::DeviceContextPool& pool =
            platform::DeviceContextPool::Instance();
        pool.Get(src_place)->Wait();
        PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(
            pimpl_->memory_copy_p2p(dst_dev_id, src_dev_id, dst, src, size));
      }
    }
  }

  void* MemoryAllocate(
      size_t dev_id, size_t size,
      MemoryAllocKind kind = MemoryAllocKind::Normal) override {
    void* ptr = nullptr;

    if (kind == MemoryAllocKind::Normal) {
      PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(
          pimpl_->device_memory_allocate(dev_id, &ptr, size));
    } else if (kind == MemoryAllocKind::Host) {
      if (!pimpl_->unified_memory_allocate) {
        PADDLE_THROW(platform::errors::Unavailable(
            "MemoryAllocKind::Host is not supported on " + Type() + "."));
      } else {
        PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(
            pimpl_->host_memory_allocate(dev_id, &ptr, size));
      }
    } else if (kind == MemoryAllocKind::Unified) {
      if (!pimpl_->unified_memory_allocate) {
        PADDLE_THROW(platform::errors::Unavailable(
            "MemoryAllocKind::Unified is not supported on " + Type() + "."));
      } else {
        PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(
            pimpl_->unified_memory_allocate(dev_id, &ptr, size));
      }
    } else {
      PADDLE_THROW(platform::errors::Unavailable("Unknow MemoryAllocKind."));
    }
    return ptr;
  }

  void MemoryDeallocate(
      size_t dev_id, void* ptr, size_t size,
      MemoryAllocKind kind = MemoryAllocKind::Normal) override {
    if (kind == MemoryAllocKind::Normal) {
      PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(
          pimpl_->device_memory_deallocate(dev_id, ptr, size));
    } else if (kind == MemoryAllocKind::Host) {
      if (!pimpl_->host_memory_deallocate) {
        PADDLE_THROW(platform::errors::Unavailable(
            "MemoryAllocKind::Host is not supported on " + Type() + "."));
      } else {
        PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(
            pimpl_->host_memory_deallocate(dev_id, ptr, size));
      }
    } else if (kind == MemoryAllocKind::Unified) {
      if (!pimpl_->unified_memory_deallocate) {
        PADDLE_THROW(platform::errors::Unavailable(
            "MemoryAllocKind::Host is not supported on " + Type() + "."));
      } else {
        PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(
            pimpl_->unified_memory_deallocate(dev_id, ptr, size));
      }
    } else {
      PADDLE_THROW(platform::errors::Unavailable("Unknow MemoryAllocKind."));
    }
  }

  void MemorySet(size_t dev_id, void* ptr, uint8_t value,
                 size_t size) override {
    if (pimpl_->device_memory_set) {
      PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(
          pimpl_->device_memory_set(dev_id, ptr, value, size));
    } else {
      void* tmp = new uint8_t[size];
      memset(tmp, value, size);
      MemoryCopy(dev_id, ptr, tmp, size, MemoryCpyKind::HostToDevice);
    }
  }

  void MemoryStats(size_t dev_id, size_t* total, size_t* free) override {
    PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(
        pimpl_->device_memory_stats(dev_id, total, free));

    size_t used = *total - *free;
    VLOG(10) << Type() + " memory usage " << (used >> 20) << "M/"
             << (*total >> 20) << "M, " << (*free >> 20)
             << "M available to allocate";
  }

  size_t GetMinChunkSize(size_t dev_id) override {
    size_t size = 0;
    pimpl_->device_min_chunk_size(dev_id, &size);
    VLOG(10) << Type() + " min chunk size " << size << "B";
    return size;
  }

  size_t GetMaxChunkSize(size_t dev_id) override {
    size_t size = 0;
    if (pimpl_->device_max_chunk_size) {
      pimpl_->device_max_chunk_size(dev_id, &size);
      VLOG(10) << Type() + " max chunk size " << size << "B";
    } else {
      return DeviceInterface::GetMaxChunkSize(dev_id);
    }
    return size;
  }

  size_t GetMaxAllocSize(size_t dev_id) override {
    size_t size = 0;
    if (pimpl_->device_max_alloc_size) {
      pimpl_->device_max_alloc_size(dev_id, &size);
      VLOG(10) << Type() + " max alloc size " << (size >> 20) << "M";
    } else {
      return DeviceInterface::GetMaxAllocSize(dev_id);
    }
    return size;
  }

  size_t GetInitAllocSize(size_t dev_id) override {
    size_t size = 0;
    if (pimpl_->device_init_alloc_size) {
      pimpl_->device_init_alloc_size(dev_id, &size);
      VLOG(10) << Type() + " init alloc size " << (size >> 20) << "M";
    } else {
      return DeviceInterface::GetInitAllocSize(dev_id);
    }
    return size;
  }

  size_t GetReallocSize(size_t dev_id) override {
    size_t size = 0;
    if (pimpl_->device_realloc_size) {
      pimpl_->device_realloc_size(dev_id, &size);
      VLOG(10) << Type() + " realloc size " << (size >> 20) << "M";
    } else {
      return DeviceInterface::GetReallocSize(dev_id);
    }
    return size;
  }

  size_t GetExtraPaddingSize(size_t dev_id) override {
    size_t padding_size = 0;
    if (pimpl_->device_extra_padding_size) {
      PADDLE_ENFORCE_PLUGGABLE_DEVICE_SUCCESS(
          pimpl_->device_extra_padding_size(dev_id, &padding_size));
      VLOG(10) << Type() + " extra padding size " << (padding_size >> 20)
               << "M";
    } else {
      return DeviceInterface::GetExtraPaddingSize(dev_id);
    }
    return 0;
  }

  size_t GetComputeCapability() override {
    size_t capability = 0;
    if (pimpl_->get_compute_capability) {
      pimpl_->get_compute_capability(&capability)
    }
    return capability;
  }

  size_t GetRuntimeVersion() override {
    size_t version = 0;
    if (pimpl_->get_runtime_version) {
      pimpl_->get_runtime_version(&version)
    }
    return version;
  }

  size_t GetDriverVersion() override {
    size_t version = 0;
    if (pimpl_->get_driver_version) {
      pimpl_->get_driver_version(&version)
    }
    return version;
  }

 private:
  inline int PlaceToIdNoCheck(const Place& place) {
    int dev_id = BOOST_GET_CONST(PluggableDevicePlace, place).GetDeviceId();
    return dev_id;
  }

  inline int PlaceToId(const Place& place) {
    int dev_id = PlaceToIdNoCheck(place);
    return dev_id;
  }
  std::unique_ptr<C_DeviceInterface> pimpl_;
  void* dso_handle_;
};

bool ValidPluggableRuntimePluginParams(const RuntimePluginParams* params) {
#define CHECK_PTR(ptr, required)                                  \
  if (params->interface->ptr == nullptr && required) {            \
    LOG(WARNING) << "DevicePlugin [type: " << params->device_type \
                 << "] pointer: " << #ptr << " is not set.";      \
    return false;                                                 \
  }

  int version = params->version.major * 10000 + params->version.minor * 100 +
                params->version.patch;
  const int platfrom_version = PADDLE_DEVICE_PLUGIN_MAJOR_VERSION * 10000 +
                               PADDLE_DEVICE_PLUGIN_MINOR_VERSION * 100 +
                               PADDLE_DEVICE_PLUGIN_PATCH_VERSION;

  if (version < platfrom_version) {
    LOG(WARNING) << "DevicePlugin [type: " << params->device_type
                 << "] version: " << version << " < PLATFROM_PLUGIN_VERSION "
                 << platfrom_version;
    return false;
  }

  CHECK_PTR(initialize, false);
  CHECK_PTR(finalize, false)

  CHECK_PTR(init_device, false);
  CHECK_PTR(set_device, true);
  CHECK_PTR(get_device, true);
  CHECK_PTR(deinit_device, false);

  CHECK_PTR(create_stream, true);
  CHECK_PTR(destroy_stream, true);
  CHECK_PTR(query_stream, false);
  CHECK_PTR(stream_add_callback, false);

  CHECK_PTR(create_event, true);
  CHECK_PTR(record_event, true);
  CHECK_PTR(destroy_event, true);
  CHECK_PTR(query_event, false);

  CHECK_PTR(synchronize_device, true);
  CHECK_PTR(synchronize_stream, true);
  CHECK_PTR(synchronize_event, true);
  CHECK_PTR(stream_wait_event, true);

  CHECK_PTR(device_memory_allocate, true);
  CHECK_PTR(device_memory_deallocate, true);
  CHECK_PTR(host_memory_allocate, false);
  CHECK_PTR(host_memory_deallocate, false);
  CHECK_PTR(unified_memory_allocate, false);
  CHECK_PTR(unified_memory_deallocate, false);
  CHECK_PTR(memory_copy_h2d, true);
  CHECK_PTR(memory_copy_d2h, true);
  CHECK_PTR(memory_copy_d2d, true);
  CHECK_PTR(memory_copy_p2p, false);
  CHECK_PTR(async_memory_copy_h2d, false);
  CHECK_PTR(async_memory_copy_d2h, false);
  CHECK_PTR(async_memory_copy_d2d, false);
  CHECK_PTR(async_memory_copy_p2p, false);

  CHECK_PTR(visible_devices_count, true);
  CHECK_PTR(visible_devices, true);
  CHECK_PTR(device_memory_stats, true);

  CHECK_PTR(device_min_chunk_size, true);
  CHECK_PTR(device_max_chunk_size, false);
  CHECK_PTR(device_max_alloc_size, false);
  CHECK_PTR(device_extra_padding_size, false);
  CHECK_PTR(get_compute_capability, false);
  CHECK_PTR(get_runtime_version, false);
  CHECK_PTR(get_driver_version, false);

  return true;
#undef CHECK_PTR
}

typedef bool (*RegisterDevicePluginFn)(RuntimePluginParams* plugin_params);

bool LoadRuntimePlugin(const RuntimePluginParams& plugin_params,
                       std::unique_ptr<C_DeviceInterface> device_interface,
                       void* dso_handle) {
  if (ValidPluggableRuntimePluginParams(&plugin_params)) {
    auto device = std::make_unique<PluggableDevice>(
        plugin_params.device_type, 255, true, std::move(device_interface),
        dso_handle);
    if (false == DeviceManager::Register(std::move(device))) {
      LOG(WARNING) << "Skip this library. Register failed!!! there may be a "
                      "Plugin with the same name.";
      return false;
      // auto plat =
      // DeviceManager::GetDeviceWithType(plugin_params.device_type);
      // if (plat) {
      //   VLOG(4) << "Visible devices count is " <<
      //   plat->VisibleDevicesCount();
      // } else {
      //   LOG(ERROR) << "Cant find DeviceInterface for " <<
      //   plugin_params.device_type;
      // }
    }
  } else {
    LOG(WARNING)
        << "Skip this library. Wrong parameters!!! please check the version "
           "compatibility between PaddlePaddle and Plugin.";
    return false;
  }
  return true;
}

bool LoadRuntimePlugin(const std::string& plugin_path) {
  RuntimePluginParams plugin_params;
  std::memset(&plugin_params, 0, sizeof(RuntimePluginParams));
  plugin_params.size = sizeof(RuntimePluginParams);
  auto device_interface = std::make_unique<C_DeviceInterface>();
  plugin_params.interface = device_interface.get();
  std::memset(plugin_params.interface, 0, sizeof(C_DeviceInterface));
  plugin_params.interface->size = sizeof(C_DeviceInterface);

  auto dso_handle = dlopen(plugin_path.c_str(), RTLD_NOW);
  RegisterDevicePluginFn init_plugin_fn =
      reinterpret_cast<RegisterDevicePluginFn>(dlsym(dso_handle, "InitPlugin"));
  if (!init_plugin_fn) {
    LOG(WARNING) << "Skip this library. InitPlugin symbol not found.";
    return false;
  }
  init_plugin_fn(&plugin_params);
  if (plugin_params.device_type == nullptr) {
    LOG(WARNING)
        << "Skip this library. InitPlugin failed!!! please check the version "
           "compatibility between PaddlePaddle and Plugin.";
    return false;
  }
  return LoadRuntimePlugin(plugin_params, std::move(device_interface),
                           dso_handle);
}

}  // namespace platform
}  // namespace paddle
