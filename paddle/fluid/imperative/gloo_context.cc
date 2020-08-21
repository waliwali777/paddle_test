//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/imperative/gloo_context.h"

namespace paddle {
namespace imperative {
#if defined(PADDLE_WITH_GLOO)
void GlooParallelContext::Init() {
  auto gloo_ptr = paddle::framework::GlooWrapper::GetInstance();
  gloo_ptr->SetRank(strategy_.rank);
  gloo_ptr->SetSize(strategy_.rank_num);
  gloo_ptr->SetPrefix(strategy_.prefix);
  gloo_ptr->SetIface(strategy_.iface);
  gloo_ptr->SetTimeoutSeconds(strategy_.init_seconds, strategy_.run_seconds);
  gloo_ptr->SetHdfsStore(strategy_.path, strategy_.fs_name, strategy_.fs_ugi);
  gloo_ptr->Init();
}
#endif

}  //  namespace imperative
}  //  namespace paddle
