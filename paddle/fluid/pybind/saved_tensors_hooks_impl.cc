// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pybind/saved_tensors_hooks_impl.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/pybind/pyobject_holder_impl.h"
#include "pybind11/pybind11.h"

#if !(defined(PADDLE_NO_PYTHON) && defined(PADDLE_ON_INFERENCE))
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#endif

namespace paddle {
namespace pybind {

#if !(defined(PADDLE_NO_PYTHON) && defined(PADDLE_ON_INFERENCE))
PackHookImpl::PackHookImpl(PyObject* hook) : hook_(hook) { Py_INCREF(hook_); }

PackHookImpl::~PackHookImpl() {
  ::pybind11::gil_scoped_acquire gil;
  Py_DECREF(hook_);
}

std::shared_ptr<PyObjectHolder> PackHookImpl::operator()(
    const paddle::experimental::Tensor& tensor) {
  bool grad_tmp = egr::Controller::Instance().HasGrad();
  egr::Controller::Instance().SetHasGrad(false);
  ::pybind11::gil_scoped_acquire gil;
  auto args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, paddle::pybind::ToPyObject(tensor));
  PyObject* ret = PyObject_Call(hook_, args, nullptr);
  PADDLE_ENFORCE_NOT_NULL(ret,
                          paddle::platform::errors::External(
                              pybind11::detail::error_string().c_str()));
  Py_XDECREF(args);
  egr::Controller::Instance().SetHasGrad(grad_tmp);
  return std::make_shared<PyObjectHolderImpl>(ret);
}

void* PackHookImpl::operator()(void* py_tensor) {
  bool grad_tmp = egr::Controller::Instance().HasGrad();
  egr::Controller::Instance().SetHasGrad(false);
  ::pybind11::gil_scoped_acquire gil;
  auto args = PyTuple_New(1);
  Py_INCREF(reinterpret_cast<PyObject*>(py_tensor));
  PyTuple_SET_ITEM(args, 0, reinterpret_cast<PyObject*>(py_tensor));
  PyObject* ret = PyObject_Call(hook_, args, nullptr);
  PADDLE_ENFORCE_NOT_NULL(ret,
                          paddle::platform::errors::External(
                              pybind11::detail::error_string().c_str()));
  Py_XDECREF(args);
  egr::Controller::Instance().SetHasGrad(grad_tmp);
  return reinterpret_cast<void*>(ret);
}

UnPackHookImpl::UnPackHookImpl(PyObject* hook) : hook_(hook) {
  Py_INCREF(hook_);
}

UnPackHookImpl::~UnPackHookImpl() {
  ::pybind11::gil_scoped_acquire gil;
  Py_DECREF(hook_);
}

paddle::experimental::Tensor UnPackHookImpl::operator()(
    std::shared_ptr<PyObjectHolder> packed_value) {
  bool grad_tmp = egr::Controller::Instance().HasGrad();
  egr::Controller::Instance().SetHasGrad(false);
  ::pybind11::gil_scoped_acquire gil;
  auto args = PyTuple_New(1);
  Py_INCREF(reinterpret_cast<PyObject*>(packed_value->get()));
  PyTuple_SET_ITEM(args, 0, reinterpret_cast<PyObject*>(packed_value->get()));
  PyObject* ret = PyObject_Call(hook_, args, nullptr);
  PADDLE_ENFORCE_NOT_NULL(ret,
                          paddle::platform::errors::External(
                              pybind11::detail::error_string().c_str()));
  Py_XDECREF(args);
  egr::Controller::Instance().SetHasGrad(grad_tmp);

  PADDLE_ENFORCE_EQ(paddle::pybind::IsEagerTensor(ret),
                    true,
                    paddle::platform::errors::InvalidArgument(
                        "paddle.autograd.saved_tensors_hooks only one pair "
                        "of hooks is allowed at a time."));

  auto tensor = reinterpret_cast<paddle::pybind::TensorObject*>(ret)->tensor;
  Py_XDECREF(ret);
  return tensor;
}

void* UnPackHookImpl::operator()(void* packed_value, void* other) {
  bool grad_tmp = egr::Controller::Instance().HasGrad();
  egr::Controller::Instance().SetHasGrad(false);
  ::pybind11::gil_scoped_acquire gil;
  auto args = PyTuple_New(1);
  Py_INCREF(reinterpret_cast<PyObject*>(packed_value));
  PyTuple_SET_ITEM(args, 0, reinterpret_cast<PyObject*>(packed_value));
  PyObject* ret = PyObject_Call(hook_, args, nullptr);
  PADDLE_ENFORCE_NOT_NULL(ret,
                          paddle::platform::errors::External(
                              pybind11::detail::error_string().c_str()));
  Py_XDECREF(args);
  egr::Controller::Instance().SetHasGrad(grad_tmp);

  PADDLE_ENFORCE_EQ(paddle::pybind::IsEagerTensor(ret),
                    true,
                    paddle::platform::errors::InvalidArgument(
                        "paddle.autograd.saved_tensors_hooks only one pair "
                        "of hooks is allowed at a time."));

  return reinterpret_cast<void*>(ret);
}
#endif

}  // namespace pybind
}  // namespace paddle
