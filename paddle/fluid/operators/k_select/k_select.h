// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "common.h"

#define MAX_THREADS 256
#define MAX_BLOCKS 128
#define MIN_COUNT_FOR_SHORT_TOPK 0
#define MIN_COUNT_FOR_LARGE_TOPK 16384
//#define MIN_COUNT_FOR_LARGE_TOPK 256000000

#define BUCKETS 10

// protocol, 0 select COO, 1 select CSR
bool k_select(float* input, int count, void* encode, void* buffer, int k,
              int protocol, cudaStream_t stream, float* moment = nullptr);

bool k_select_bucket(float* input, int count, void* encode, void* buffer, int k,
                     int protocol, cudaStream_t stream,
                     float* moment = nullptr);

int get_buffer_size(int count);

bool k_select(float* input, int input_count, void* encode, int k,
              cudaStream_t stream);
