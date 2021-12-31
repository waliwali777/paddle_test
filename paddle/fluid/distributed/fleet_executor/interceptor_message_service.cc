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
#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE) && \
    !defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/distributed/fleet_executor/interceptor_message_service.h"
#include "brpc/server.h"
#include "paddle/fluid/distributed/fleet_executor/carrier.h"
#include "paddle/fluid/distributed/fleet_executor/global_map.h"

namespace paddle {
namespace distributed {

void InterceptorMessageServiceImpl::InterceptorMessageService(
    google::protobuf::RpcController* control_base,
    const InterceptorMessage* request, InterceptorResponse* response,
    google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  VLOG(3) << "Interceptor Message Service receives a message from interceptor "
          << request->src_id() << " to interceptor " << request->dst_id()
          << ", with the message: " << request->message_type();
  // TODO(liyurui): Remove this hard code.
  int64_t carrier_id;
  if (request->ctrl_message()) {
    carrier_id = 0;
  } else {
    carrier_id = *GlobalMap<int64_t, int64_t>::Get(request->dst_id());
  }
  bool flag = GlobalMap<int64_t, Carrier>::Get(carrier_id)
                  ->EnqueueInterceptorMessage(*request);
  response->set_rst(flag);
}

}  // namespace distributed
}  // namespace paddle
#endif
