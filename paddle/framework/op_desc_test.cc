/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/operator.h"
#include "gtest/gtest.h"

TEST(Operator, Create) {
  using paddle::framework::Operator;
  paddle::framework::OpDesc op_desc;
  op_desc.set_type("ADD");
  op_desc.add_inputs("X");
  op_desc.add_inputs("Y");
  op_desc.add_outputs("Z");

  auto attr = op_desc.mutable_attrs()->Add();
  attr->set_type(paddle::framework::AttrType::FLOAT);
  attr->set_f(3.14);

  auto op = new Operator(op_desc);
}