/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <stdint.h>
#include <sys/stat.h>
#include <ostream>
#include <thread>

#include <unistd.h>
#include <iostream>

#include "paddle/framework/data_type.h"
#include "paddle/framework/executor.h"
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/detail/send_recv_impl.h"
#include "paddle/operators/detail/simple_block_queue.h"

namespace paddle {
namespace operators {

void RunServer(std::shared_ptr<detail::SendRecvServerImpl> service,
               const std::string &server_address) {
  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(service.get());
  std::unique_ptr<Server> server(builder.BuildAndStart());
  LOG(INFO) << "Server listening on " << server_address << std::endl;
  server->Wait();
}

class RecvOp : public framework::OperatorBase {
 public:
  RecvOp(const std::string &type, const framework::VariableNameMap &inputs,
         const framework::VariableNameMap &outputs,
         const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {
    if (!rpc_service_) {
      rpc_service_.reset(new detail::SendRecvServerImpl());
      std::string endpoint = Attr<std::string>("endpoint");
      server_thread_.reset(new std::thread(RunServer, rpc_service_, endpoint));
    }
  }

  virtual ~RecvOp() { server_thread_->join(); }

  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    // blocking get one var from client.
    std::cout << "before get from client..." << std::endl;
    const framework::LoDTensor &t = rpc_service_->Get();
    framework::Scope &recv_scope = scope.NewScope();
    // set graph input var
    auto *var = recv_scope.FindVar(Input("X"));
    auto *tensor = var->GetMutable<framework::LoDTensor>();
    // FIXME(typhoonzero): do not copy
    tensor->CopyFrom(t, dev_ctx.GetPlace(), dev_ctx);

    auto *block = Attr<framework::BlockDescBind *>("OptimizeBlock");
    auto *program = block->Program();
    framework::Executor executor(dev_ctx);
    // Run sub graph to get optimized tensor
    executor.Run(*program, &recv_scope, block->ID(),
                 false /*create_local_scope*/);

    auto *out_var = recv_scope.FindVar("Out");
    // push back
    rpc_service_->Push(out_var->Get<framework::LoDTensor>());
  }

 protected:
  std::shared_ptr<detail::SendRecvServerImpl> rpc_service_;
  std::shared_ptr<std::thread> server_thread_;
};

class RecvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  RecvOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(Tensor) Input tensor to be saved");
    AddComment(R"DOC(
Recv operator

This operator will recv tensor from send_op
)DOC");
    AddAttr<std::string>("endpoint",
                         "(string, default 127.0.0.1:6164)"
                         "IP address to listen on.")
        .SetDefault("127.0.0.1:6164")
        .AddCustomChecker([](const std::string &ip) { return !ip.empty(); });
    AddAttr<framework::BlockDescBind *>("OptimizeBlock", "type BlockDescBind*",
                                        "optimize network run in server");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(recv, ops::RecvOp, ops::RecvOpMaker);
