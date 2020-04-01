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

#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/string/string_helper.h"
#include "paddle/fluid/platform/cuda_device_guard.h"

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

namespace paddle {
namespace framework {

void HeterTask::PackTask(Scope* thread_scope, int taskid, DataFeed* reader) {
  
  auto &block = program_.Block(0);
  scope_ = thread_scope->NewScope();
  for (auto &var : block.AllVars()) {
    if (!var->Persistable()) {
      auto *ptr = scope_->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
    }
  }
  state_ = PULL_SPARSE;
  auto& use_slots = reader->use_slots_;
  for (size_t i = 0; i < use_slots.size(); ++i) {
    Variable* thread_var = thread_scope->FindVar(use_slots[i])->GetMutable<LoDTensor>();
    LoDTensor* thread_tensor = thread_var->GetMutable<LoDTensor>();
    Variable* task_var = thread_scope->FindVar(use_slots[i])->GetMutable<LoDTensor>();
    LoDTensor* task_tensor = task_var->GetMutable<LoDTensor>();
    TensorCopy(*thread_tensor, platform::CPUPlace(), task_tensor);
  }

}

void HeterCpuWorker::Schedule(int taskid) {
  auto task = wait_queue_->Get(taskid);
  if (task) {
    run_queue_->Put(task->taskid_, task);
  }
}

void HeterCpuWorker::JumpContext(std::shared_ptr<HeterTask> task) {
  if (!(wait_queue_->TryPut(task->taskid_, task))) {
    run_queue_->Put(task);
  }
}

void HeterCpuWorker::Initialize(const TrainerDesc& desc) {
  param_ = desc.downpour_param();
  for (int i = 0; i < param_.sparse_table_size(); ++i) {
    uint64_t table_id =
        static_cast<uint64_t>(param_.sparse_table(i).table_id());
    TableParameter table = param_.sparse_table(i);
    sparse_key_names_[table_id].resize(table.sparse_key_name_size());
    for (int j = 0; j < table.sparse_key_name_size(); ++j) {
      sparse_key_names_[table_id][j] = table.sparse_key_name(j);
    }
    sparse_value_names_[table_id].resize(table.sparse_value_name_size());
    for (int j = 0; j < table.sparse_value_name_size(); ++j) {
      sparse_value_names_[table_id][j] = table.sparse_value_name(j);
    }
    sparse_grad_names_[table_id].resize(table.sparse_grad_name_size());
    for (int j = 0; j < table.sparse_grad_name_size(); ++j) {
      sparse_grad_names_[table_id][j] = table.sparse_grad_name(j);
    }
    label_var_name_[table_id] = table.label_var_name();
    sparse_push_keys_[table_id] = std::vector<uint64_t>();
  }

  for (int i = 0; i < param_.dense_table_size(); ++i) {
    uint64_t table_id = static_cast<uint64_t>(param_.dense_table(i).table_id());
    auto table = param_.dense_table(i);
    dense_value_names_[table_id].resize(table.dense_value_name_size());
    for (int j = 0; j < table.dense_value_name_size(); ++j) {
      dense_value_names_[table_id][j] = table.dense_value_name(j);
    }
    dense_grad_names_[table_id].resize(table.dense_grad_name_size());
    for (int j = 0; j < table.dense_grad_name_size(); ++j) {
      dense_grad_names_[table_id][j] = table.dense_grad_name(j);
    }
  }

  skip_ops_.resize(param_.skip_ops_size());
  for (int i = 0; i < param_.skip_ops_size(); ++i) {
    skip_ops_[i] = param_.skip_ops(i);
  }
  for (int i = 0; i < param_.stat_var_names_size(); ++i) {
    stat_var_name_map_[param_.stat_var_names(i)] = 1;
  }

  need_to_push_sparse_ = param_.push_sparse();
  need_to_push_dense_ = param_.push_dense();

  fleet_ptr_ = FleetWrapper::GetInstance();
  fetch_config_ = desc.fetch_config();
  use_cvm_ = desc.use_cvm();
  // for sparse value accessor, embedding only
  no_cvm_ = desc.no_cvm();
  scale_datanorm_ = desc.scale_datanorm();
  dump_slot_ = desc.dump_slot();
  dump_fields_.resize(desc.dump_fields_size());
  for (int i = 0; i < desc.dump_fields_size(); ++i) {
    dump_fields_[i] = desc.dump_fields(i);
  }
  adjust_ins_weight_config_ = desc.adjust_ins_weight_config();
  need_dump_param_ = false;
  dump_param_.resize(desc.dump_param_size());
  for (int i = 0; i < desc.dump_param_size(); ++i) {
    dump_param_[i] = desc.dump_param(i);
  }
  if (desc.dump_param_size() != 0) {
    need_dump_param_ = true;
  }
  for (int i = 0; i < desc.check_nan_var_names_size(); ++i) {
    check_nan_var_names_.push_back(desc.check_nan_var_names(i));
  }
  copy_table_config_ = desc.copy_table_config();
  for (int i = 0; i < copy_table_config_.src_sparse_tables_size(); ++i) {
    uint64_t src_table = copy_table_config_.src_sparse_tables(i);
    uint64_t dest_table = copy_table_config_.dest_sparse_tables(i);
    VLOG(3) << "copy_sparse_tables_ push back " << src_table << "->"
            << dest_table;
    copy_sparse_tables_.push_back(std::make_pair(src_table, dest_table));
  }
  for (int i = 0; i < copy_table_config_.src_dense_tables_size(); ++i) {
    uint64_t src_table = copy_table_config_.src_dense_tables(i);
    uint64_t dest_table = copy_table_config_.dest_dense_tables(i);
    VLOG(3) << "copy_dense_tables_ push back " << src_table << "->"
            << dest_table;
    copy_dense_tables_.push_back(std::make_pair(src_table, dest_table));
  }
  for (auto& m : copy_table_config_.table_denpendency_map()) {
    if (sparse_key_names_.find(m.key()) != sparse_key_names_.end()) {
      // currently only support one dependency
      for (auto& value : m.values()) {
        table_dependency_[m.key()] = value;
      }
    }
  }
}

void HeterCpuWorker::SetChannelWriter(ChannelObject<std::string>* queue) {
  writer_.Reset(queue);
}

void HeterCpuWorker::SetNeedDump(bool need_dump_field) {
  need_dump_field_ = need_dump_field;
}

void HeterCpuWorker::CreateEvent() {
  #ifdef PADDLE_WITH_CUDA
  auto dev_id = boost::get<platform::CUDAPlace>(place_).device;
  platform::CUDADeviceGuard guard(dev_id);
  PADDLE_ENFORCE(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
  #endif
}

void HeterCpuWorker::CreatePinVar() {
  #ifdef PADDLE_WITH_CUDA
  for (auto& v : sparse_value_names_) {
    for (auto& name : v.second) {
      auto *ptr = thread_scope_->Var(name + "pin");
      InitializeVariable(ptr, proto::VarType::LOD_TENSOR);
    }
  }
  for (auto& v : sparse_grad_names_) {
    for (auto& name : v.second) {
      auto *ptr = thread_scope_->Var(name + "pin");
      InitializeVariable(ptr, proto::VarType::LOD_TENSOR);
    }
  }
  for (auto& v : dense_grad_names_) {
    for (auto& name : v.second) {
      auto *ptr = thread_scope_->Var(name + "pin");
      InitializeVariable(ptr, proto::VarType::LOD_TENSOR);
    }
  }
  #endif
}

template <typename T>
std::string PrintLodTensorType(LoDTensor* tensor, int64_t start, int64_t end) {
  auto count = tensor->numel();
  if (start < 0 || end > count) {
    VLOG(3) << "access violation";
    return "access violation";
  }
  std::ostringstream os;
  for (int64_t i = start; i < end; i++) {
    os << ":" << tensor->data<T>()[i];
  }
  return os.str();
}

std::string PrintLodTensorIntType(LoDTensor* tensor, int64_t start,
                                  int64_t end) {
  auto count = tensor->numel();
  if (start < 0 || end > count) {
    VLOG(3) << "access violation";
    return "access violation";
  }
  std::ostringstream os;
  for (int64_t i = start; i < end; i++) {
    os << ":" << static_cast<uint64_t>(tensor->data<int64_t>()[i]);
  }
  return os.str();
}

std::string PrintLodTensor(LoDTensor* tensor, int64_t start, int64_t end) {
  std::string out_val;
  if (tensor->type() == proto::VarType::FP32) {
    out_val = PrintLodTensorType<float>(tensor, start, end);
  } else if (tensor->type() == proto::VarType::INT64) {
    out_val = PrintLodTensorIntType(tensor, start, end);
  } else if (tensor->type() == proto::VarType::FP64) {
    out_val = PrintLodTensorType<double>(tensor, start, end);
  } else {
    out_val = "unsupported type";
  }
  return out_val;
}

std::pair<int64_t, int64_t> GetTensorBound(LoDTensor* tensor, int index) {
  auto& dims = tensor->dims();
  if (tensor->lod().size() != 0) {
    auto& lod = tensor->lod()[0];
    return {lod[index] * dims[1], lod[index + 1] * dims[1]};
  } else {
    return {index * dims[1], (index + 1) * dims[1]};
  }
}

bool CheckValidOutput(LoDTensor* tensor, size_t batch_size) {
  auto& dims = tensor->dims();
  if (dims.size() != 2) return false;
  if (tensor->lod().size() != 0) {
    auto& lod = tensor->lod()[0];
    if (lod.size() != batch_size + 1) {
      return false;
    }
  } else {
    if (dims[0] != static_cast<int>(batch_size)) {
      return false;
    }
  }
  return true;
}

void HeterCpuWorker::DumpParam() {
  std::string os;
  for (auto& param : dump_param_) {
    os.clear();
    os = param;
    Variable* var = thread_scope_->FindVar(param);
    if (var == nullptr) {
      continue;
    }
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    int64_t len = tensor->numel();
    os += PrintLodTensor(tensor, 0, len);
    writer_ << os;
  }
}

void HeterCpuWorker::CollectLabelInfo(HeterTask* task, size_t table_idx) {
  if (no_cvm_) {
    return;
  }
  uint64_t table_id = static_cast<uint64_t>(
      param_.program_config(0).pull_sparse_table_id(table_idx));

  TableParameter table;
  for (auto i : param_.sparse_table()) {
    if (i.table_id() == table_id) {
      table = i;
      break;
    }
  }
  auto& feature = (task->features_)[table_id];
  auto& feature_label = (task->feature_labels_)[table_id];
  feature_label.resize(feature.size());
  Variable* var = thread_scope_->FindVar(label_var_name_[table_id]);
  LoDTensor* tensor = var->GetMutable<LoDTensor>();
  int64_t* label_ptr = tensor->data<int64_t>();

  size_t global_index = 0;
  for (size_t i = 0; i < sparse_key_names_[table_id].size(); ++i) {
    VLOG(3) << "sparse_key_names_[" << i
            << "]: " << sparse_key_names_[table_id][i];
    Variable* fea_var = thread_scope_->FindVar(sparse_key_names_[table_id][i]);
    if (fea_var == nullptr) {
      continue;
    }
    LoDTensor* tensor = fea_var->GetMutable<LoDTensor>();
    CHECK(tensor != nullptr) << "tensor of var "
                             << sparse_key_names_[table_id][i] << " is null";

    // skip slots which do not have embedding
    Variable* emb_var =
        thread_scope_->FindVar(sparse_value_names_[table_id][i]);
    if (emb_var == nullptr) {
      continue;
    }

    int64_t* ids = tensor->data<int64_t>();
    size_t fea_idx = 0;
    // tensor->lod()[0].size() == batch_size + 1
    for (auto lod_idx = 1u; lod_idx < tensor->lod()[0].size(); ++lod_idx) {
      for (; fea_idx < tensor->lod()[0][lod_idx]; ++fea_idx) {
        // should be skipped feasign defined in protobuf
        if (ids[fea_idx] == 0u) {
          continue;
        }
        feature_label[global_index++] =
            static_cast<float>(label_ptr[lod_idx - 1]);
      }
    }
  }
  CHECK(global_index == feature.size())
      << "expect fea info size:" << feature.size() << " real:" << global_index;
}

void HeterCpuWorker::FillSparseValue(HeterTask* task, size_t table_idx) {
  uint64_t table_id = static_cast<uint64_t>(
      param_.program_config(0).pull_sparse_table_id(table_idx));

  TableParameter table;
  for (auto i : param_.sparse_table()) {
    if (i.table_id() == table_id) {
      table = i;
      break;
    }
  }

  auto& fea_value = (task->feature_values_)[table_id];
  auto fea_idx = 0u;

  std::vector<float> init_value(table.fea_dim());
  for (size_t i = 0; i < sparse_key_names_[table_id].size(); ++i) {
    std::string slot_name = sparse_key_names_[table_id][i];
    std::string emb_slot_name = sparse_value_names_[table_id][i];
    Variable* var = thread_scope_->FindVar(slot_name);
    if (var == nullptr) {
      continue;
    }
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    CHECK(tensor != nullptr) << "tensor of var " << slot_name << " is null";
    int64_t* ids = tensor->data<int64_t>();
    int len = tensor->numel();
    Variable* var_emb = thread_scope_->FindVar(emb_slot_name);
    if (var_emb == nullptr) {
      continue;
    }
    LoDTensor* tensor_emb = var_emb->GetMutable<LoDTensor>();
    float* ptr = tensor_emb->mutable_data<float>({len, table.emb_dim()},
                                                 place_);
    //memset(ptr, 0, sizeof(float) * len * table.emb_dim());
    auto& tensor_lod = tensor->lod()[0];
    LoD data_lod{tensor_lod};
    tensor_emb->set_lod(data_lod);

    #ifdef PADDLE_WITH_CUDA
    Variable* pin_var = thread_scope_->FindVar(emb_slot_name + "pin");
    LoDTensor* pin_tensor = pin_var->GetMutable<LoDTensor>();
    ptr = pin_tensor->mutable_data<float>({len, table.emb_dim()},
                                                 platform::CUDAPinnedPlace());
    #endif

    bool is_nid = (adjust_ins_weight_config_.need_adjust() &&
                   adjust_ins_weight_config_.nid_slot() == emb_slot_name);
    if (is_nid) {
      nid_show_.clear();
    }
    int nid_ins_index = 0;

    for (int index = 0; index < len; ++index) {
      if (use_cvm_ || no_cvm_) {
        if (ids[index] == 0u) {
          memcpy(ptr + table.emb_dim() * index, init_value.data(),
                 sizeof(float) * table.emb_dim());
          if (is_nid) {
            nid_show_.push_back(-1);
            ++nid_ins_index;
          }
          continue;
        }
        memcpy(ptr + table.emb_dim() * index, fea_value[fea_idx].data(),
               sizeof(float) * table.emb_dim());
        if (is_nid &&
            static_cast<size_t>(index) == tensor->lod()[0][nid_ins_index]) {
          nid_show_.push_back(fea_value[fea_idx][0]);
          ++nid_ins_index;
        }
        fea_idx++;
      } else {
        if (ids[index] == 0u) {
          memcpy(ptr + table.emb_dim() * index, init_value.data() + 2,
                 sizeof(float) * table.emb_dim());
          if (is_nid) {
            nid_show_.push_back(-1);
            ++nid_ins_index;
          }
          continue;
        }
        memcpy(ptr + table.emb_dim() * index, fea_value[fea_idx].data() + 2,
               sizeof(float) * table.emb_dim());
        if (is_nid &&
            static_cast<size_t>(index) == tensor->lod()[0][nid_ins_index]) {
          nid_show_.push_back(fea_value[fea_idx][0]);
          ++nid_ins_index;
        }
        fea_idx++;
      }
    }
    #ifdef PADDLE_WITH_CUDA
    {
      float* ptr = tensor_emb->data<float>();
      float* pin_ptr = pin_tensor->data<float>();
      memory::Copy(
          boost::get<platform::CUDAPlace>(place_),
          ptr,
          platform::CUDAPinnedPlace(),
          pin_ptr,
          len * table.emb_dim() * sizeof(float), copy_stream_);
    }
    #endif
  }
}

void HeterCpuWorker::AdjustInsWeight() {
#ifdef _LINUX
  // check var and tensor not null
  if (!adjust_ins_weight_config_.need_adjust()) {
    VLOG(0) << "need_adjust=false, skip adjust ins weight";
    return;
  }
  Variable* nid_var =
      thread_scope_->FindVar(adjust_ins_weight_config_.nid_slot());
  if (nid_var == nullptr) {
    VLOG(0) << "nid slot var " << adjust_ins_weight_config_.nid_slot()
            << " is nullptr, skip adjust ins weight";
    return;
  }
  LoDTensor* nid_tensor = nid_var->GetMutable<LoDTensor>();
  if (nid_tensor == nullptr) {
    VLOG(0) << "tensor of nid slot var " << adjust_ins_weight_config_.nid_slot()
            << " is nullptr, skip adjust ins weight";
    return;
  }
  Variable* ins_weight_var =
      thread_scope_->FindVar(adjust_ins_weight_config_.ins_weight_slot());
  if (ins_weight_var == nullptr) {
    VLOG(0) << "ins weight var " << adjust_ins_weight_config_.ins_weight_slot()
            << " is nullptr, skip adjust ins weight";
    return;
  }
  LoDTensor* ins_weight_tensor = ins_weight_var->GetMutable<LoDTensor>();
  if (ins_weight_tensor == nullptr) {
    VLOG(0) << "tensor of ins weight tensor "
            << adjust_ins_weight_config_.ins_weight_slot()
            << " is nullptr, skip adjust ins weight";
    return;
  }

  float* ins_weights = ins_weight_tensor->data<float>();
  size_t len = ins_weight_tensor->numel();  // len = batch size
  // here we assume nid_show slot only has one feasign in each instance
  CHECK(len == nid_show_.size()) << "ins_weight size should be equal to "
                                 << "nid_show size, " << len << " vs "
                                 << nid_show_.size();
  float nid_adjw_threshold = adjust_ins_weight_config_.nid_adjw_threshold();
  float nid_adjw_ratio = adjust_ins_weight_config_.nid_adjw_ratio();
  int64_t nid_adjw_num = 0;
  double nid_adjw_weight = 0.0;
  size_t ins_index = 0;
  for (size_t i = 0; i < len; ++i) {
    float nid_show = nid_show_[i];
    VLOG(3) << "nid_show " << nid_show;
    if (nid_show < 0) {
      VLOG(3) << "nid_show < 0, continue";
      continue;
    }
    float ins_weight = 1.0;
    if (nid_show >= 0 && nid_show < nid_adjw_threshold) {
      ins_weight = log(M_E +
                       (nid_adjw_threshold - nid_show) / nid_adjw_threshold *
                           nid_adjw_ratio);
      // count nid adjw insnum and weight
      ++nid_adjw_num;
      nid_adjw_weight += ins_weight;
      // choose large ins weight
      VLOG(3) << "ins weight new " << ins_weight << ", ins weight origin "
              << ins_weights[ins_index];
      if (ins_weight > ins_weights[ins_index]) {
        VLOG(3) << "ins " << ins_index << " weight changes to " << ins_weight;
        ins_weights[ins_index] = ins_weight;
      }
      ++ins_index;
    }
  }
  VLOG(3) << "nid adjw info: total_adjw_num: " << nid_adjw_num
          << ", avg_adjw_weight: " << nid_adjw_weight;
#endif
}

void HeterCpuWorker::CopySparseTable() {
  for (size_t i = 0; i < copy_sparse_tables_.size(); ++i) {
    int64_t src_table = copy_sparse_tables_[i].first;
    int64_t dest_table = copy_sparse_tables_[i].second;
    int32_t feanum = 0;
    if (src_table == dest_table) {
      continue;
    } else if (!copy_table_config_.sparse_copy_by_feasign()) {
      if (feasign_set_.find(src_table) == feasign_set_.end()) {
        continue;
      } else if (feasign_set_[src_table].size() == 0) {
        continue;
      }
      feanum = fleet_ptr_->CopyTable(src_table, dest_table);
    } else {
      std::vector<uint64_t> fea_vec(feasign_set_[src_table].begin(),
                                    feasign_set_[src_table].end());
      feanum = fleet_ptr_->CopyTableByFeasign(src_table, dest_table, fea_vec);
      fea_vec.clear();
      std::vector<uint64_t>().swap(fea_vec);
    }
    VLOG(3) << "copy feasign from table " << src_table << " to table "
            << dest_table << ", feasign num=" << feanum;
    feasign_set_[src_table].clear();
    std::unordered_set<uint64_t>().swap(feasign_set_[src_table]);
  }
  feasign_set_.clear();
}

void HeterCpuWorker::CopyDenseTable() {
  if (thread_id_ != 0) {
    return;
  }
  thread_local std::vector<std::future<int32_t>> pull_dense_status;
  for (size_t i = 0; i < copy_dense_tables_.size(); ++i) {
    uint64_t src_table = copy_dense_tables_[i].first;
    uint64_t dest_table = copy_dense_tables_[i].second;
    if (src_table == dest_table) {
      continue;
    }
    int32_t dim = fleet_ptr_->CopyTable(src_table, dest_table);
    VLOG(3) << "copy param from table " << src_table << " to table "
            << dest_table << ", dim=" << dim;
    if (copy_table_config_.dense_pull_after_copy()) {
      VLOG(3) << "dense pull after copy, table=" << dest_table;
      pull_dense_status.resize(0);
      //fleet_ptr_->PullDenseVarsAsync(*root_scope_, dest_table,
      //                               dense_value_names_[dest_table],
      //                               &pull_dense_status);
      for (auto& t : pull_dense_status) {
        t.wait();
        auto status = t.get();
        if (status != 0) {
          LOG(WARNING) << "pull dense after copy table failed,"
                       << " table=" << dest_table;
        }
      }
    }
  }
}

void HeterCpuWorker::CreateThreadParam(const ProgramDesc& program) {
  #ifdef PADDLE_WITH_CUDA
  auto dev_id = boost::get<platform::CUDAPlace>(place_).device;
  platform::CUDADeviceGuard guard(dev_id);
  auto &block = program.Block(0);
  for (auto& var : block.AllVars()) {
    if (var->Persistable()) {
      auto name = var->Name();
      Variable* root_var = root_scope_->FindVar(name);
      LoDTensor* root_tensor = root_var->GetMutable<LoDTensor>();
      auto *ptr = thread_scope_->Var(name);
      InitializeVariable(ptr, proto::VarType::LOD_TENSOR);
      LoDTensor* thread_tensor = ptr->GetMutable<LoDTensor>();

#define MemcpyCallback(cpp_type, proto_type)                              \
  do {                                                                    \
    if (root_tensor->type() == proto_type) {                              \
      MemCpy<cpp_type>(thread_tensor, root_tensor, place_, copy_stream_); \
    }                                                                     \
  } while (0)
        _ForEachDataType_(MemcpyCallback);

    }
  }
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaEventRecord(event_, copy_stream_));
  cudaEventSynchronize(event_);
  #endif
}

template <typename T>
void HeterCpuWorker::MemCpy(LoDTensor *thread_tensor, LoDTensor *root_tensor,
                            const paddle::platform::Place& thread_place, 
                            cudaStream_t stream) {
  T* thread_ptr = thread_tensor->mutable_data<T>(root_tensor->dims(), thread_place);
  T* root_ptr = root_tensor->data<T>();
  if (platform::is_cpu_place(root_tensor->place())) {
    memory::Copy(
        boost::get<platform::CUDAPlace>(thread_place),
        thread_ptr,
        platform::CPUPlace(),
        root_ptr, sizeof(T) * root_tensor->numel(), stream);
  }
  else {
    memory::Copy(
        boost::get<platform::CUDAPlace>(thread_place),
        thread_ptr,
        boost::get<platform::CUDAPlace>(root_tensor->place()),
        root_ptr, sizeof(T) * root_tensor->numel(), stream);
  }
}

void HeterCpuWorker::CopyDenseVars() {
  if (thread_id_ != 0) {
    return;
  }
  for (int i = 0; i < copy_table_config_.src_var_list_size(); ++i) {
    auto& src_var_name = copy_table_config_.src_var_list(i);
    auto& dest_var_name = copy_table_config_.dest_var_list(i);
    if (src_var_name == dest_var_name) {
      continue;
    }
    VLOG(3) << "copy dense var from " << src_var_name << " to "
            << dest_var_name;
    Variable* src_var = thread_scope_->FindVar(src_var_name);
    CHECK(src_var != nullptr) << src_var_name << " not found";  // NOLINT
    LoDTensor* src_tensor = src_var->GetMutable<LoDTensor>();
    CHECK(src_tensor != nullptr) << src_var_name
                                 << " tensor is null";  // NOLINT
    float* src_data = src_tensor->data<float>();

    Variable* dest_var = thread_scope_->FindVar(dest_var_name);
    CHECK(dest_var != nullptr) << dest_var_name << " not found";  // NOLINT
    LoDTensor* dest_tensor = dest_var->GetMutable<LoDTensor>();
    CHECK(dest_tensor != nullptr) << dest_var_name
                                  << " tensor is null";  // NOLINT
    float* dest_data = dest_tensor->data<float>();

    CHECK(src_tensor->numel() == dest_tensor->numel())
        << "tensor numel not equal," << src_tensor->numel() << " vs "
        << dest_tensor->numel();
    for (int i = 0; i < src_tensor->numel(); i++) {
      dest_data[i] = src_data[i];
    }
  }
}

/*
void HeterCpuWorker::TrainFilesWithProfiler() {
  auto dev_id = boost::get<platform::CUDAPlace>(place_).device;
  platform::SetDeviceId(dev_id);
  VLOG(3) << "Begin to train files with profiler";
  platform::SetNumThreads(1);
  device_reader_->Start();
  std::vector<double> op_total_time;
  std::vector<std::string> op_name;
  for (auto& op : ops_) {
    bool need_skip = false;
    for (auto t = 0u; t < skip_ops_.size(); ++t) {
      if (op->Type().find(skip_ops_[t]) != std::string::npos) {
        need_skip = true;
        break;
      }
    }
    if (!need_skip) {
      op_name.push_back(op->Type());
    }
  }

  VLOG(3) << "op name size: " << op_name.size();
  op_total_time.resize(op_name.size());
  for (size_t i = 0; i < op_total_time.size(); ++i) {
    op_total_time[i] = 0.0;
  }
  platform::Timer timeline;
  double total_time = 0.0;
  double read_time = 0.0;
  double pull_sparse_time = 0.0;
  double adjust_ins_weight_time = 0.0;
  double collect_label_time = 0.0;
  double fill_sparse_time = 0.0;
  double push_sparse_time = 0.0;
  double push_dense_time = 0.0;
  double copy_table_time = 0.0;
  int cur_batch;
  int batch_cnt = 0;
  uint64_t total_inst = 0;
  timeline.Start();
  while ((cur_batch = device_reader_->Next()) > 0) {
    timeline.Pause();
    read_time += timeline.ElapsedSec();
    total_time += timeline.ElapsedSec();

    timeline.Start();
    if (copy_table_config_.need_copy()) {
      VLOG(3) << "copy_sparse_tables_.size " << copy_sparse_tables_.size();
      if (copy_table_config_.sparse_copy_by_feasign()) {
        for (size_t i = 0; i < copy_sparse_tables_.size(); ++i) {
          uint64_t tid = copy_sparse_tables_[i].first;
          feasign_set_[tid].insert(sparse_push_keys_[tid].begin(),
                                   sparse_push_keys_[tid].end());
        }
      }
      if (batch_cnt % copy_table_config_.batch_num() == 0) {
        CopySparseTable();
        CopyDenseTable();
        CopyDenseVars();
      }
    }
    timeline.Pause();
    copy_table_time += timeline.ElapsedSec();
    total_time += timeline.ElapsedSec();

    VLOG(3) << "program config size: " << param_.program_config_size();
    for (int i = 0; i < param_.program_config(0).pull_sparse_table_id_size();
         ++i) {
      uint64_t tid = static_cast<uint64_t>(
          param_.program_config(0).pull_sparse_table_id(i));
      TableParameter table;
      for (auto j : param_.sparse_table()) {
        if (j.table_id() == tid) {
          table = j;
          break;
        }
      }
      timeline.Start();
      fleet_ptr_->PullSparseVarsSync(
          *thread_scope_, tid, sparse_key_names_[tid], &features_[tid],
          &feature_values_[tid], table.fea_dim(), sparse_value_names_[tid]);
      timeline.Pause();
      pull_sparse_time += timeline.ElapsedSec();
      total_time += timeline.ElapsedSec();
      timeline.Start();
      CollectLabelInfo(i);
      timeline.Pause();
      collect_label_time += timeline.ElapsedSec();
      total_time += timeline.ElapsedSec();
      timeline.Start();
      FillSparseValue(i);
      timeline.Pause();
      fill_sparse_time += timeline.ElapsedSec();
      total_time += timeline.ElapsedSec();
      timeline.Start();
      auto nid_iter = std::find(sparse_value_names_[tid].begin(),
                                sparse_value_names_[tid].end(),
                                adjust_ins_weight_config_.nid_slot());
      if (nid_iter != sparse_value_names_[tid].end()) {
        AdjustInsWeight();
      }
      timeline.Pause();
      adjust_ins_weight_time += timeline.ElapsedSec();
      total_time += timeline.ElapsedSec();
    }
    VLOG(3) << "Fill sparse value for all sparse table done.";

    #ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaEventRecord(event_, copy_stream_));
    cudaEventSynchronize(event_);
    #endif
    int run_op_idx = 0;
    for (auto& op : ops_) {
      bool need_skip = false;
      for (auto t = 0u; t < skip_ops_.size(); ++t) {
        if (op->Type().find(skip_ops_[t]) != std::string::npos) {
          need_skip = true;
          break;
        }
      }
      if (!need_skip) {
        timeline.Start();
        VLOG(3) << "Going to run op " << op_name[run_op_idx];
        op->Run(*thread_scope_, place_);
        VLOG(3) << "Op " << op_name[run_op_idx] << " Finished";
        timeline.Pause();
        op_total_time[run_op_idx++] += timeline.ElapsedSec();
        total_time += timeline.ElapsedSec();
      }
    }
    #ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaEventRecord(event_, copy_stream_));
    #endif
    // check inf and nan
    for (std::string& var_name : check_nan_var_names_) {
      Variable* var = thread_scope_->FindVar(var_name);
      if (var == nullptr) {
        continue;
      }
      LoDTensor* tensor = var->GetMutable<LoDTensor>();
      if (tensor == nullptr) {
        continue;
      }
      PADDLE_ENFORCE_EQ(framework::TensorContainsInf(*tensor), false,
                        "Tensor %s contains Inf", var_name);
      PADDLE_ENFORCE_EQ(framework::TensorContainsNAN(*tensor), false,
                        "Tensor %s contains NAN", var_name);
    }

    if (need_to_push_sparse_) {
      for (int i = 0; i < param_.program_config(0).push_sparse_table_id_size();
           ++i) {
        uint64_t tid = static_cast<uint64_t>(
            param_.program_config(0).push_sparse_table_id(i));
        TableParameter table;
        for (auto i : param_.sparse_table()) {
          if (i.table_id() == tid) {
            table = i;
            break;
          }
        }
        timeline.Start();
        #ifdef PADDLE_WITH_CUDA
        fleet_ptr_->PushSparseVarsWithLabelAsync(
            *thread_scope_, tid, features_[tid], feature_labels_[tid],
            sparse_key_names_[tid], sparse_grad_names_[tid], table.emb_dim(),
            &feature_grads_[tid], &push_sparse_status_, cur_batch, use_cvm_,
            dump_slot_, &sparse_push_keys_[tid], no_cvm_, place_, copy_stream_, event_);
        #else
        fleet_ptr_->PushSparseVarsWithLabelAsync(
            *thread_scope_, tid, features_[tid], feature_labels_[tid],
            sparse_key_names_[tid], sparse_grad_names_[tid], table.emb_dim(),
            &feature_grads_[tid], &push_sparse_status_, cur_batch, use_cvm_,
            dump_slot_, &sparse_push_keys_[tid], no_cvm_);
        #endif
        timeline.Pause();
        push_sparse_time += timeline.ElapsedSec();
        total_time += timeline.ElapsedSec();
      }
    }

    if (need_to_push_dense_) {
      timeline.Start();
      for (int i = 0; i < param_.program_config(0).push_dense_table_id_size();
           ++i) {
        uint64_t tid = static_cast<uint64_t>(
            param_.program_config(0).push_dense_table_id(i));
        #ifdef PADDLE_WITH_CUDA
        fleet_ptr_->PushDenseVarsAsync(
            *thread_scope_, tid, dense_grad_names_[tid], &push_sparse_status_,
            scale_datanorm_, cur_batch, place_, copy_stream_, event_);
        #else
        fleet_ptr_->PushDenseVarsAsync(
            *thread_scope_, tid, dense_grad_names_[tid], &push_sparse_status_,
            scale_datanorm_, cur_batch);
        #endif
      }
      timeline.Pause();
      push_dense_time += timeline.ElapsedSec();
      total_time += timeline.ElapsedSec();
      VLOG(3) << "push sparse and dense gradient done.";
      int32_t tmp_push_dense_wait_times = -1;
      static uint32_t push_dense_wait_times =
          static_cast<uint32_t>(tmp_push_dense_wait_times);
      if (push_dense_status_.size() >= push_dense_wait_times) {
        for (auto& t : push_dense_status_) {
          t.wait();
        }
        push_dense_status_.resize(0);
      }

      if (tmp_push_dense_wait_times == -1) {
        push_dense_status_.resize(0);
      }
    }

    if (need_to_push_sparse_) {
      int32_t tmp_push_sparse_wait_times = -1;
      static uint32_t push_sparse_wait_times =
          static_cast<uint32_t>(tmp_push_sparse_wait_times);
      if (push_sparse_status_.size() >= push_sparse_wait_times) {
        for (auto& t : push_sparse_status_) {
          t.wait();
        }
        push_sparse_status_.resize(0);
      }

      if (tmp_push_sparse_wait_times == -1) {
        push_sparse_status_.resize(0);
      }

      VLOG(3) << "going to increase thread version";
      VLOG(3) << "push dense table id size: "
              << param_.program_config(0).push_dense_table_id_size();
    }

    if (need_to_push_dense_) {
      for (int i = 0; i < param_.program_config(0).push_dense_table_id_size();
           ++i) {
        uint64_t tid = static_cast<uint64_t>(
            param_.program_config(0).push_dense_table_id(i));
        pull_dense_worker_->IncreaseThreadVersion(thread_id_, tid);
      }
    }

    PrintFetchVars();
    thread_scope_->DropKids();
    total_inst += cur_batch;
    ++batch_cnt;

    if (thread_id_ == 0) {
      // should be configured here
      if (batch_cnt > 0 && batch_cnt % 100 == 0) {
        double op_sum_time = 0;
        std::unordered_map<std::string, double> op_to_time;
        for (size_t i = 0; i < op_total_time.size(); ++i) {
          fprintf(stderr, "op_name:[%zu][%s], op_mean_time:[%fs]\n", i,
                  op_name[i].c_str(), op_total_time[i] / batch_cnt);
          if (op_to_time.find(op_name[i]) == op_to_time.end()) {
            op_to_time[op_name[i]] = 0.0;
          }
          op_to_time[op_name[i]] += op_total_time[i];
          op_sum_time += op_total_time[i];
        }
        for (auto& i : op_to_time) {
          fprintf(stderr, "op [%s] run total time: [%f]ms\n", i.first.c_str(),
                  i.second / batch_cnt);
        }
        fprintf(stderr, "op run total time: %fs\n", op_sum_time / batch_cnt);
        fprintf(stderr, "train total time: %fs\n", total_time / batch_cnt);
        fprintf(stderr, "pull sparse time: %fs\n",
                pull_sparse_time / batch_cnt);
        fprintf(stderr, "fill sparse time: %fs\n",
                fill_sparse_time / batch_cnt);
        fprintf(stderr, "push sparse time: %fs\n",
                push_sparse_time / batch_cnt);
        fprintf(stderr, "push dense time: %fs\n", push_dense_time / batch_cnt);
        fprintf(stderr, "collect label time: %fs\n",
                collect_label_time / batch_cnt);
        fprintf(stderr, "adjust ins weight time: %fs\n",
                adjust_ins_weight_time / batch_cnt);
        fprintf(stderr, "copy table time: %fs\n", copy_table_time / batch_cnt);
        fprintf(stderr, "mean read time: %fs\n", read_time / batch_cnt);
        fprintf(stderr, "IO percent: %f\n", read_time / total_time * 100);
        fprintf(stderr, "op run percent: %f\n", op_sum_time / total_time * 100);
        fprintf(stderr, "pull sparse time percent: %f\n",
                pull_sparse_time / total_time * 100);
        fprintf(stderr, "adjust ins weight time percent: %f\n",
                adjust_ins_weight_time / total_time * 100);
        fprintf(stderr, "copy table time percent: %f\n",
                copy_table_time / total_time * 100);
        fprintf(stderr, "collect label time percent: %f\n",
                collect_label_time / total_time * 100);
        fprintf(stderr, "fill sparse time percent: %f\n",
                fill_sparse_time / total_time * 100);
        fprintf(stderr, "push sparse time percent: %f\n",
                push_sparse_time / total_time * 100);
        fprintf(stderr, "push dense time percent: %f\n",
                push_dense_time / total_time * 100);
        fprintf(stderr, "%6.2f instances/s\n", total_inst / total_time);
      }
    }
    timeline.Start();
  }
  if (copy_table_config_.need_copy()) {
    CopySparseTable();
    CopyDenseTable();
    CopyDenseVars();
  }
}
*/

void HeterCpuWorker::TrainFiles() {
  auto dev_id = boost::get<platform::CUDAPlace>(place_).device;
  platform::SetDeviceId(dev_id);
  VLOG(3) << "Begin to train files";
  platform::SetNumThreads(1);
  device_reader_->Start();
  int batch_cnt = 0;
  int cur_batch;
  //while ((cur_batch = device_reader_->Next()) > 0) {
  while (1) {
    //if (copy_table_config_.need_copy()) {
    //  if (copy_table_config_.sparse_copy_by_feasign()) {
    //    for (size_t i = 0; i < copy_sparse_tables_.size(); ++i) {
    //      uint64_t tid = copy_sparse_tables_[i].first;
    //      feasign_set_[tid].insert(sparse_push_keys_[tid].begin(),
    //                               sparse_push_keys_[tid].end());
    //    }
    //  }
    //  if (batch_cnt % copy_table_config_.batch_num() == 0) {
    //    CopySparseTable();
    //    CopyDenseTable();
    //    CopyDenseVars();
    //  }
    //}

    std::shared_ptr<HeterTask> task;
    if (!run_queue_.empty()) {
      task = run_queue_.get();
    }
    else {
      cur_batch = device_reader_->Next();
      if (cur_batch < 0) {
        break;
      }
      int taskid = batch_cnt * worker_num_ + thread_id_;
      task = object_pool_.Get();
      task->PackTask(thread_scope_, taskid, device_reader_);
    }
    for (;;) {
      // pull sparse here
      if (task->state == PULL_SPARSE) {
        for (int i = 0; i < param_.program_config(0).pull_sparse_table_id_size();
             ++i) {
          uint64_t tid = static_cast<uint64_t>(
              param_.program_config(0).pull_sparse_table_id(i));
          TableParameter table;
          for (auto j : param_.sparse_table()) {
            if (j.table_id() == tid) {
              table = j;
              break;
            }
          }
          fleet_ptr_->HeterPullSparseVars(
              task, tid, sparse_key_names_[tid],
              table.fea_dim(), sparse_value_names_[tid]);
        }
        JumpContext(task);
      }
      else if (task->state == OP_RUN) {
        for (int i = 0; i < param_.program_config(0).pull_sparse_table_id_size();
             ++i) {
          CollectLabelInfo(task, i);
          FillSparseValue(task, i);
        }
        auto nid_iter = std::find(sparse_value_names_[tid].begin(),
                                  sparse_value_names_[tid].end(),
                                  adjust_ins_weight_config_.nid_slot());
        if (nid_iter != sparse_value_names_[tid].end()) {
          AdjustInsWeight();
        }
        
        VLOG(3) << "fill sparse value for all sparse table done.";
        // do computation here
        for (auto& op : ops_) {
          bool need_skip = false;
          for (auto t = 0u; t < skip_ops_.size(); ++t) {
            if (op->Type().find(skip_ops_[t]) != std::string::npos) {
              need_skip = true;
              break;
            }
          }
          if (!need_skip) {
            op->Run(task->scope_, place_);
          }
        }
        // check inf and nan
        for (std::string& var_name : check_nan_var_names_) {
          Variable* var = thread_scope_->FindVar(var_name);
          if (var == nullptr) {
            continue;
          }
          LoDTensor* tensor = var->GetMutable<LoDTensor>();
          if (tensor == nullptr) {
            continue;
          }
          PADDLE_ENFORCE_EQ(framework::TensorContainsInf(*tensor), false,
                            "Tensor %s contains Inf", var_name);
          PADDLE_ENFORCE_EQ(framework::TensorContainsNAN(*tensor), false,
                            "Tensor %s contains NAN", var_name);
        }
        task->update();
      }
      else if (task->state == PUSH_GRAD) {
        if (need_to_push_sparse_) {
          // push gradients here
          for (int i = 0; i < param_.program_config(0).push_sparse_table_id_size();
               ++i) {
            uint64_t tid = static_cast<uint64_t>(
                param_.program_config(0).push_sparse_table_id(i));
            TableParameter table;
            for (auto i : param_.sparse_table()) {
              if (i.table_id() == tid) {
                table = i;
                break;
              }
            }
            fleet_ptr_->HeterPushSparseVars(
                task, tid,
                sparse_key_names_[tid], sparse_grad_names_[tid], table.emb_dim(),
                &push_sparse_status_, cur_batch, use_cvm_,
                dump_slot_, no_cvm_);
          }
        }

        if (need_to_push_sparse_) {
          VLOG(3) << "push sparse gradient done.";
          int32_t tmp_push_sparse_wait_times = -1;
          static uint32_t push_sparse_wait_times =
              static_cast<uint32_t>(tmp_push_sparse_wait_times);
          if (push_sparse_status_.size() >= push_sparse_wait_times) {
            for (auto& t : push_sparse_status_) {
              t.wait();
            }
            push_sparse_status_.resize(0);
          }

          if (tmp_push_sparse_wait_times == -1) {
            push_sparse_status_.resize(0);
          }
        }

        if (need_dump_field_) {
          size_t batch_size = device_reader_->GetCurBatchSize();
          std::vector<std::string> ars(batch_size);
          for (auto& ar : ars) {
            ar.clear();
          }
          auto& ins_id_vec = device_reader_->GetInsIdVec();
          auto& ins_content_vec = device_reader_->GetInsContentVec();
          for (size_t i = 0; i < ins_id_vec.size(); i++) {
            ars[i] += ins_id_vec[i];
            ars[i] = ars[i] + "\t" + ins_content_vec[i];
          }
          for (auto& field : dump_fields_) {
            Variable* var = thread_scope_->FindVar(field);
            if (var == nullptr) {
              continue;
            }
            LoDTensor* tensor = var->GetMutable<LoDTensor>();
            if (!CheckValidOutput(tensor, batch_size)) {
              continue;
            }
            for (size_t i = 0; i < batch_size; ++i) {
              auto output_dim = tensor->dims()[1];
              std::string output_dimstr =
                  boost::lexical_cast<std::string>(output_dim);
              ars[i] = ars[i] + "\t" + field + ":" + output_dimstr;
              auto bound = GetTensorBound(tensor, i);
              ars[i] += PrintLodTensor(tensor, bound.first, bound.second);
            }
          }
          // #pragma omp parallel for
          for (size_t i = 0; i < ars.size(); i++) {
            if (ars[i].length() == 0) {
              continue;
            }
            writer_ << ars[i];
          }
          if (need_dump_param_ && thread_id_ == 0) {
            DumpParam();
          }
        }

        PrintFetchVars();
        thread_scope_->DropKids();
        ++batch_cnt;
        task->update();
      }
      else if (task->state == DONE) {
        object_pool_->push(task);
        break;
      }
  }
  if (need_dump_field_) {
    writer_.Flush();
  }
  if (copy_table_config_.need_copy()) {
    CopySparseTable();
    CopyDenseTable();
    CopyDenseVars();
  }
}

}  // end namespace framework
}  // end namespace paddle
