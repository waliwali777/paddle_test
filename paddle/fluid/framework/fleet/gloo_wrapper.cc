/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#include <vector>
#include "paddle/fluid/framework/io/fs.h"
#include "paddle/fluid/platform/errors.h"

namespace gloo {
namespace rendezvous {

HdfsStore::HdfsStore(const std::string& path) {
  path_ = path;
  wait_sleep_ms_ = 3000;
  wait_timeout_ = std::chrono::seconds(999999999);
  retry_times_ = 100;
}

void HdfsStore::set(const std::string& key, const std::vector<char>& data) {
#ifdef PADDLE_WITH_GLOO
  auto tmp = TmpPath(key);
  auto path = ObjectPath(key);
  bool is_exists = paddle::framework::fs_exists(path);
  if (is_exists) {
    LOG(WARNING) << "path exists, will be removed: " << path;
    paddle::framework::fs_remove(path);
  }
  int err_no = 0;
  for (int i = 1; i <= retry_times_; ++i) {
    err_no = 0;
    std::shared_ptr<FILE> fp =
        paddle::framework::fs_open_write(tmp, &err_no, "");
    size_t write_count = fwrite_unlocked(data.data(), 1, data.size(), fp.get());
    if (write_count != data.size()) {
      VLOG(0) << "fwrite_unlocked failed, retry times " << i << " write_count "
              << write_count << " data.size() " << data.size();
      fp.reset();
      sleep(wait_sleep_ms_ / 1000);
      continue;
    }
    fp.reset();
    if (err_no != 0) {
      VLOG(0) << "fs_open_write failed, retry times " << i << " err no "
              << err_no;
      sleep(wait_sleep_ms_ / 1000);
      continue;
    }
    break;
  }
  paddle::framework::fs_mv(tmp, path);
#endif
}

std::vector<char> HdfsStore::get(const std::string& key) {
  auto path = ObjectPath(key);
  std::vector<char> result;
#ifdef PADDLE_WITH_GLOO
  // block until key is set
  wait({key});
  bool is_exists = paddle::framework::fs_exists(path);
  PADDLE_ENFORCE_EQ(is_exists, true,
                    paddle::platform::errors::NotFound(
                        "HdfsStore::get, path not exists: " + path));
  int err_no = 0;
  std::shared_ptr<FILE> fp = paddle::framework::fs_open_read(path, &err_no, "");
  char buffer = '\0';
  size_t read_count = 0;
  while (fread(&buffer, 1, 1, fp.get()) == 1) {
    ++read_count;
    result.push_back(buffer);
  }
  VLOG(3) << "HdfsStore::get read_count " << read_count;
#endif
  return result;
}

void HdfsStore::wait(const std::vector<std::string>& keys) {
#ifdef PADDLE_WITH_GLOO
  wait(keys, wait_timeout_);  // NOLINT
#endif
}

void HdfsStore::wait(const std::vector<std::string>& keys,
                     const std::chrono::milliseconds&) {  // NOLINT
#ifdef PADDLE_WITH_GLOO
  auto start = std::chrono::steady_clock::now();
  int last_check_rank = -1;
  while (!Check(keys, &last_check_rank)) {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start);
    if (wait_timeout_ != gloo::kNoTimeout && elapsed > wait_timeout_) {
      PADDLE_THROW("TIMEOUT self_rank = " + std::to_string(self_rank_)
                   + " pair_rank = " +  std::to_string(last_check_rank));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_sleep_ms_));
  }
#endif
}

void HdfsStore::SetTimeoutSeconds(int timeout_seconds) {
  wait_timeout_ = std::chrono::seconds(timeout_seconds);
}

std::string HdfsStore::EncodeName(const std::string& name) {
  thread_local std::hash<std::string> hash_func;
  return std::to_string(hash_func(name));
}

std::string HdfsStore::TmpPath(const std::string& name) {
  return path_ + "/" + EncodeName(name) + "_tmp";
}

std::string HdfsStore::ObjectPath(const std::string& name) {
  return path_ + "/" + EncodeName(name);
}

bool HdfsStore::Check(const std::vector<std::string>& keys,
                      int* last_check_rank) {
#ifdef PADDLE_WITH_GLOO
  std::vector<std::string> paths;
  for (const auto& key : keys) {
    paths.push_back(ObjectPath(key));
  }
  int index = -1;
  for (const auto& path : paths) {
    ++index;
    bool is_exists = paddle::framework::fs_exists(path);
    VLOG(3) << "HdfsStore::Check " << is_exists << " path " << path;
    if (!is_exists) {
      if (last_check_rank) {
        if (index >= self_rank_) { ++index; }
        *last_check_rank = index;
      }
      return false;
    }
  }
  if (last_check_rank) {
    *last_check_rank = -1;
  }
#endif
  return true;
}

}  // namespace rendezvous
}  // namespace gloo

namespace paddle {
namespace framework {

void GlooWrapper::Init() {
  if (is_initialized_) {
    return;
  }
  std::string cmd = std::string("${HADOOP_HOME}/bin/hadoop fs");
  cmd += " -D fs.default.name=" + hdfs_name_;
  cmd += " -D hadoop.job.ugi=" + hdfs_ugi_;
  paddle::framework::hdfs_set_command(cmd);
#ifdef PADDLE_WITH_GLOO
  gloo::transport::tcp::attr attr;
  attr.iface = iface_;
  std::shared_ptr<gloo::rendezvous::HdfsStore> file_store = nullptr;
  switch (store_type_) {
    case GlooStoreType::HDFS: {
      file_store = std::make_shared<gloo::rendezvous::HdfsStore>(hdfs_path_);
      file_store->SetTimeoutSeconds(init_timeout_.count());
      file_store->SetRank(rank_);
      prefix_store_ =
          std::make_shared<gloo::rendezvous::PrefixStore>(prefix_, *file_store);
      break;
    }
    default:
      PADDLE_THROW("unknown store type: " + std::to_string(store_type_));
  }
  CHECK(prefix_store_ != nullptr) << "store should not be null";  // NOLINT
  auto dev = gloo::transport::tcp::CreateDevice(attr);
  auto context = std::make_shared<gloo::rendezvous::Context>(rank_, size_);
  context->setTimeout(wait_timeout_);
  context->connectFullMesh(*prefix_store_, dev);
  context_ = std::move(context);
#endif
  is_initialized_ = true;
}

template std::vector<int64_t> GlooWrapper::AllReduce<int64_t>(
    std::vector<int64_t>& sendbuf,  // NOLINT
    const std::string& mode);
template std::vector<float> GlooWrapper::AllReduce<float>(
    std::vector<float>& sendbuf,  // NOLINT
    const std::string& mode);
template std::vector<double> GlooWrapper::AllReduce<double>(
    std::vector<double>& sendbuf,  // NOLINT
    const std::string& mode);
template std::vector<uint64_t> GlooWrapper::AllReduce<uint64_t>(
    std::vector<uint64_t>& sendbuf,  // NOLINT
    const std::string& mode);
template std::vector<int64_t> GlooWrapper::AllGather<int64_t>(
    int64_t& input);  // NOLINT
template std::vector<uint64_t> GlooWrapper::AllGather<uint64_t>(
    uint64_t& input);  // NOLINT
template std::vector<float> GlooWrapper::AllGather<float>(
    float& input);  // NOLINT
template std::vector<double> GlooWrapper::AllGather<double>(
    double& input);  // NOLINT

}  // namespace framework
}  // namespace paddle
