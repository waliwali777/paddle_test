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

#include <numeric>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <class bidiiter>
bidiiter sample_unique(bidiiter begin, bidiiter end, int num_samples) {
  size_t left = std::distance(begin, end);
  for (int i = 0; i < num_samples; i++) {
    bidiiter r = begin;
    std::advance(r, rand() % left);
    std::swap(*begin, *r);
    ++begin;
    --left;
  }
  return begin;
}

template <typename T>
void sample_neighbors(const T* src, const T* dst_count, std::vector<T>* inputs,
                      std::vector<T>* outputs, std::vector<T>* output_counts,
                      int k) {
  const size_t bs = inputs->size();
  output_counts->resize(bs);
  outputs->resize(k * bs);

  size_t total_neighbors = 0;
  for (size_t i = 0; i < bs; i++) {
    T node = inputs->data()[i];
    T begin = dst_count[node];
    T end = dst_count[node + 1];
    int cap = end - begin;
    if (k < cap) {
      std::vector<T> cut_src(cap);
      std::copy(src + begin, src + end, cut_src.begin());
      sample_unique(cut_src.begin(), cut_src.end(), k);
      std::copy(cut_src.begin(), cut_src.begin() + k,
                outputs->data() + total_neighbors);
      cap = k;
    } else {
      std::copy(src + begin, src + end, outputs->data() + total_neighbors);
    }
    total_neighbors += cap;
    *(output_counts->data() + i) = cap;
  }
  outputs->resize(total_neighbors);

  std::vector<T> unique_outputs(outputs->size());
  std::sort(inputs->begin(), inputs->end());
  std::vector<T> outputs_sort(outputs->size());
  std::copy(outputs->begin(), outputs->end(), outputs_sort.begin());
  std::sort(outputs_sort.begin(), outputs_sort.end());
  auto unique_outputs_end = std::set_difference(
      outputs_sort.begin(), outputs_sort.end(), inputs->begin(), inputs->end(),
      unique_outputs.begin());
  unique_outputs_end = std::unique(unique_outputs.begin(), unique_outputs_end);
  inputs->resize(std::distance(unique_outputs.begin(), unique_outputs_end));
  std::copy(unique_outputs.begin(), unique_outputs_end, inputs->begin());
}

template <typename DeviceContext, typename T>
class GraphSampleOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // 1. Get inputs.
    auto* src = ctx.Input<Tensor>("Src");
    auto* dst_count = ctx.Input<Tensor>("Dst_Count");
    auto* vertices = ctx.Input<Tensor>("X");
    std::vector<int> sample_sizes = ctx.Attr<std::vector<int>>("sample_sizes");

    const T* src_data = src->data<T>();
    const T* dst_count_data = dst_count->data<T>();
    const T* p_vertices = vertices->data<T>();
    const size_t bs = vertices->dims()[0];

    // 2. Sample neighbors.
    std::vector<T> inputs(bs);
    std::vector<T> outputs;
    std::vector<T> output_counts;
    std::copy(p_vertices, p_vertices + bs, inputs.begin());
    std::vector<std::vector<T>> dst_vec;
    dst_vec.push_back(inputs);
    std::vector<std::vector<T>> outputs_vec;
    std::vector<std::vector<T>> output_counts_vec;

    const size_t num_layers = sample_sizes.size();
    for (size_t i = 0; i < num_layers; i++) {
      if (inputs.size() == 0) {
        break;
      }
      if (i > 0) {
        dst_vec.push_back(inputs);
      }
      sample_neighbors<T>(src_data, dst_count_data, &inputs, &outputs,
                          &output_counts, sample_sizes[i]);
      outputs_vec.push_back(outputs);
      output_counts_vec.push_back(output_counts);
    }

    // 3. Concat intermediate sample results
    int64_t unique_dst_size = 0, src_size = 0;
    for (size_t i = 0; i < num_layers; i++) {
      unique_dst_size += dst_vec[i].size();
      src_size += outputs_vec[i].size();
    }
    std::vector<T> unique_dst_merge(unique_dst_size);
    std::vector<T> src_merge(src_size);
    std::vector<T> dst_sample_counts_merge(unique_dst_size);
    auto unique_dst_merge_ptr = unique_dst_merge.begin();
    auto src_merge_ptr = src_merge.begin();
    auto dst_sample_counts_merge_ptr = dst_sample_counts_merge.begin();
    for (size_t i = 0; i < num_layers; i++) {
      if (i == 0) {
        unique_dst_merge_ptr = std::copy(dst_vec[i].begin(), dst_vec[i].end(),
                                         unique_dst_merge.begin());
        src_merge_ptr = std::copy(outputs_vec[i].begin(), outputs_vec[i].end(),
                                  src_merge.begin());
        dst_sample_counts_merge_ptr =
            std::copy(output_counts_vec[i].begin(), output_counts_vec[i].end(),
                      dst_sample_counts_merge.begin());
      } else {
        unique_dst_merge_ptr = std::copy(dst_vec[i].begin(), dst_vec[i].end(),
                                         unique_dst_merge_ptr);
        src_merge_ptr = std::copy(outputs_vec[i].begin(), outputs_vec[i].end(),
                                  src_merge_ptr);
        dst_sample_counts_merge_ptr =
            std::copy(output_counts_vec[i].begin(), output_counts_vec[i].end(),
                      dst_sample_counts_merge_ptr);
      }
    }

    int64_t num_sample_edges = std::accumulate(
        dst_sample_counts_merge.begin(), dst_sample_counts_merge.end(), 0);
    PADDLE_ENFORCE_EQ(
        src_merge.size(), num_sample_edges,
        platform::errors::External("Number of sample edges dismatch."));

    // 4. Reindex.
    std::unordered_map<T, T> node_map;
    std::vector<T> unique_nodes;
    size_t reindex_id = 0;
    for (size_t i = 0; i < unique_dst_merge.size(); i++) {
      T node = unique_dst_merge[i];
      unique_nodes.push_back(node);
      node_map[node] = reindex_id++;
    }
    for (size_t i = 0; i < src_merge.size(); i++) {
      T node = src_merge[i];
      if (node_map.find(node) == node_map.end()) {
        unique_nodes.push_back(node);
        node_map[node] = reindex_id++;
      }
      src_merge[i] = node_map[node];
    }
    std::vector<T> dst_merge(src_merge.size());
    size_t cnt = 0;
    for (size_t i = 0; i < unique_dst_merge.size(); i++) {
      for (T j = 0; j < dst_sample_counts_merge[i]; j++) {
        T node = unique_dst_merge[i];
        dst_merge[cnt++] = node_map[node];
      }
    }

    // 5. Get outputs.
    auto* sample_index = ctx.Output<Tensor>("Sample_index");
    auto* out_src = ctx.Output<Tensor>("Out_Src");
    auto* out_dst = ctx.Output<Tensor>("Out_Dst");
    sample_index->Resize({static_cast<int>(unique_nodes.size())});
    out_src->Resize({static_cast<int>(src_merge.size())});
    out_dst->Resize({static_cast<int>(src_merge.size())});
    T* p_sample_index = sample_index->mutable_data<T>(ctx.GetPlace());
    T* p_out_src = out_src->mutable_data<T>(ctx.GetPlace());
    T* p_out_dst = out_dst->mutable_data<T>(ctx.GetPlace());
    const size_t& sample_bytes = unique_nodes.size() * sizeof(T);
    memset(p_sample_index, 0, sample_bytes);
    std::copy(unique_nodes.begin(), unique_nodes.end(), p_sample_index);
    const size_t& memset_bytes = src_merge.size() * sizeof(T);
    memset(p_out_src, 0, memset_bytes);
    memset(p_out_dst, 0, memset_bytes);
    std::copy(src_merge.begin(), src_merge.end(), p_out_src);
    std::copy(dst_merge.begin(), dst_merge.end(), p_out_dst);
  }
};

}  // namespace operators
}  // namespace paddle
