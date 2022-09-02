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

/**
 * We now still need TensorWrapper and it is designed to Copy
 * tensor in autograd mode.
 *
 * Since in autograd usage, we need to pass autograd_meta to
 * backward computation however in tensor interface add to much
 * autograd_related method is not a good choice.
 *
 * In TensorWrapper we will keep autograd info to backward, only
 * for input var, but for output var it will only copy autograd
 * with no grad **/

#include "paddle/fluid/eager/tensor_wrapper.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/saved_tensors_hooks.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#if !(!defined(WITH_PYTHON) && defined(ON_INFER))
#include "paddle/fluid/pybind/eager.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#endif

#if !(!defined(WITH_PYTHON) && defined(ON_INFER))
PyTypeObject* p_tensor_type_tensor_wrapper = nullptr;
#endif
namespace egr {

#if !(!defined(WITH_PYTHON) && defined(ON_INFER))
PyObject* ToPyObject(const paddle::experimental::Tensor& value) {
  PyObject* obj = nullptr;
  obj = p_tensor_type_tensor_wrapper->tp_alloc(p_tensor_type_tensor_wrapper, 0);
  if (obj) {
    auto v = reinterpret_cast<paddle::pybind::TensorObject*>(obj);
    new (&(v->tensor)) paddle::experimental::Tensor();
    v->tensor = value;
  } else {
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "tp_alloc return null, can not new a PyObject."));
  }
  return obj;
}
#endif
TensorWrapper::TensorWrapper(const TensorWrapper& other) {
  no_need_buffer_ = other.no_need_buffer_;
  intermidiate_tensor_ = other.intermidiate_tensor_;
  weak_grad_node_ = other.weak_grad_node_;
  inplace_version_snapshot_ = other.inplace_version_snapshot_;
  packed_tensor_info_ = other.packed_tensor_info_;
  unpack_hook_ = other.unpack_hook_;
  Py_XINCREF(packed_tensor_info_);
  Py_XINCREF(unpack_hook_);
}

TensorWrapper& TensorWrapper::operator=(const TensorWrapper& other) {
  no_need_buffer_ = other.no_need_buffer_;
  intermidiate_tensor_ = other.intermidiate_tensor_;
  weak_grad_node_ = other.weak_grad_node_;
  inplace_version_snapshot_ = other.inplace_version_snapshot_;
  packed_tensor_info_ = other.packed_tensor_info_;
  unpack_hook_ = other.unpack_hook_;
  Py_XINCREF(packed_tensor_info_);
  Py_XINCREF(unpack_hook_);
  return *this;
}

TensorWrapper::TensorWrapper(const paddle::experimental::Tensor& tensor,
                             bool no_need_buffer) {
  // set inplace_version_snapshot_ according to tensor's current inplace
  // version.
  if (tensor.impl() && phi::DenseTensor::classof(tensor.impl().get())) {
    phi::DenseTensor* dense_tensor =
        static_cast<phi::DenseTensor*>(tensor.impl().get());
    auto& inplace_version_counter = dense_tensor->InplaceVersionCounter();
    inplace_version_snapshot_ = inplace_version_counter.CurrentVersion();
  }

  /**
   * Normally, we should only save data and part of autograd_meta of fwd
   * tensor, and should not reserve its original grad_node,
   * to avoid recursive and additional depends on GradNodeBase
   * **/
  auto* tensor_autograd_meta = EagerUtils::nullable_autograd_meta(tensor);
  no_need_buffer_ = no_need_buffer;
  // shallow copy tensor_impl here
  if (no_need_buffer) {
    if (phi::DenseTensor::classof(tensor.impl().get())) {
      // Only Copy Meta
      phi::DenseTensor* dense_tensor =
          static_cast<phi::DenseTensor*>(tensor.impl().get());
      // TODO(jiabin): It's not a good idea to set memory size to zero, find
      // another way and change this.
      intermidiate_tensor_.set_impl(
          std::move(std::make_shared<phi::DenseTensor>(
              std::make_shared<phi::Allocation>(nullptr, 0, tensor.place()),
              std::move(dense_tensor->meta()))));
    } else {
      PADDLE_THROW(paddle::platform::errors::Fatal(
          "Unrecognized tensor type for no_need_buffer feature"));
    }
  } else {
#if !(!defined(WITH_PYTHON) && defined(ON_INFER))
    if (SavedTensorsHooks::GetInstance().is_enable() &&
        tensor.is_dense_tensor()) {
      phi::DenseTensor* dense_tensor =
          static_cast<phi::DenseTensor*>(tensor.impl().get());
      intermidiate_tensor_.set_impl(
          std::move(std::make_shared<phi::DenseTensor>(
              std::make_shared<phi::Allocation>(nullptr, 0, tensor.place()),
              dense_tensor->meta())));
      auto pack_hook = SavedTensorsHooks::GetInstance().get_pack_hook();
      unpack_hook_ = SavedTensorsHooks::GetInstance().get_unpack_hook();
      Py_XINCREF(unpack_hook_);
      bool grad_tmp = egr::Controller::Instance().HasGrad();
      egr::Controller::Instance().SetHasGrad(false);
      auto args = PyTuple_New(1);
      auto obj = ToPyObject(tensor);
      Py_INCREF(obj);
      PyTuple_SET_ITEM(args, 0, obj);
      ::pybind11::gil_scoped_acquire gil;
      packed_tensor_info_ = PyObject_Call(pack_hook, args, nullptr);
      Py_XDECREF(args);
      egr::Controller::Instance().SetHasGrad(grad_tmp);
    } else {
#endif
      intermidiate_tensor_.set_impl(tensor.impl());
#if !(!defined(WITH_PYTHON) && defined(ON_INFER))
    }
#endif
  }

  if (VLOG_IS_ON(7)) {
    // TODO(jiabin): This may has server performance issue
    intermidiate_tensor_.set_name(tensor.name() + "@Saved");
  }

  if (tensor_autograd_meta) {
    auto autograd_meta = std::make_shared<AutogradMeta>(*tensor_autograd_meta);
    autograd_meta->ResetGradNode();
    intermidiate_tensor_.set_autograd_meta(autograd_meta);
    weak_grad_node_ = tensor_autograd_meta->GetMutableGradNode();
  }
}

TensorWrapper::~TensorWrapper() {
  Py_XDECREF(packed_tensor_info_);
  Py_XDECREF(unpack_hook_);
}
paddle::experimental::Tensor TensorWrapper::recover() {
  VLOG(6) << "Recover tensor: " << intermidiate_tensor_.name()
          << " for wrapper";
  if (!intermidiate_tensor_.defined()) {
    VLOG(6) << "Return NULL tensor Here. ";
    return paddle::experimental::Tensor();
  }

#if !(!defined(WITH_PYTHON) && defined(ON_INFER))
  if (packed_tensor_info_ && unpack_hook_) {
    bool grad_tmp = egr::Controller::Instance().HasGrad();
    egr::Controller::Instance().SetHasGrad(false);
    PyObject* py_tensor = nullptr;
    {
      auto args = PyTuple_New(1);
      Py_INCREF(packed_tensor_info_);
      PyTuple_SET_ITEM(args, 0, packed_tensor_info_);
      ::pybind11::gil_scoped_acquire gil;
      py_tensor = PyObject_Call(unpack_hook_, args, nullptr);
      Py_XDECREF(args);
    }
    if (p_tensor_type_tensor_wrapper &&
        PyObject_IsInstance(
            py_tensor,
            reinterpret_cast<PyObject*>(p_tensor_type_tensor_wrapper))) {
      auto src_dense_tensor = static_cast<phi::DenseTensor*>(
          reinterpret_cast<paddle::pybind::TensorObject*>(py_tensor)
              ->tensor.impl()
              .get());
      static_cast<phi::DenseTensor*>(intermidiate_tensor_.impl().get())
          ->ResetHolder(src_dense_tensor->MoveMemoryHolder());
      Py_XDECREF(py_tensor);
    } else {
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "The unpack_hook mast return a Tensor."));
    }
    egr::Controller::Instance().SetHasGrad(grad_tmp);
  } else {
#endif
    check_inplace_version();
#if !(!defined(WITH_PYTHON) && defined(ON_INFER))
  }
#endif

  paddle::experimental::Tensor recovered_tensor = intermidiate_tensor_;

  std::shared_ptr<GradNodeBase> new_grad_node = weak_grad_node_.lock();
  if (new_grad_node) {
    VLOG(3) << "Recovered TensorWrapper with GradNode " << new_grad_node->name()
            << " addr: " << new_grad_node.get();
  } else {
    VLOG(3) << "Recovered TensorWrapper with Empty GradNode";
  }
  auto* intermediate_autograd_meta =
      EagerUtils::nullable_autograd_meta(intermidiate_tensor_);

  if (intermediate_autograd_meta) {
    auto p_ab_autograd_meta =
        std::make_shared<AutogradMeta>(*intermediate_autograd_meta);
    if (new_grad_node) {
      p_ab_autograd_meta->SetGradNode(new_grad_node);
    }
    recovered_tensor.set_autograd_meta(p_ab_autograd_meta);
  }

  return recovered_tensor;
}

paddle::experimental::Tensor TensorWrapper::get_intermidiate_tensor() {
  return intermidiate_tensor_;
}

void TensorWrapper::clear() { intermidiate_tensor_.reset(); }

void TensorWrapper::check_inplace_version() {
  if (no_need_buffer_) {
    VLOG(6) << "There's no need to check inplace_version because "
               "no_need_buffer_ is true.";
    return;
  }
  if (intermidiate_tensor_.impl() &&
      phi::DenseTensor::classof(intermidiate_tensor_.impl().get())) {
    phi::DenseTensor* dense_tensor =
        static_cast<phi::DenseTensor*>(intermidiate_tensor_.impl().get());
    auto& inplace_version_counter = dense_tensor->InplaceVersionCounter();

    uint32_t wrapper_version_snapshot = inplace_version_snapshot_;
    uint32_t tensor_version = inplace_version_counter.CurrentVersion();
    PADDLE_ENFORCE_EQ(
        tensor_version,
        wrapper_version_snapshot,
        paddle::platform::errors::PermissionDenied(
            "Tensor '%s' used in gradient computation has been "
            "modified by an inplace operation. "
            "Its version is %d but the expected version is %d. "
            "Please fix your code to void calling an inplace operator "
            "after using the Tensor which will used in gradient "
            "computation.",
            intermidiate_tensor_.name(),
            tensor_version,
            wrapper_version_snapshot));
    VLOG(6) << " The wrapper_version_snapshot of Tensor '"
            << intermidiate_tensor_.name() << "' is [ "
            << wrapper_version_snapshot << " ]";
    VLOG(6) << " The tensor_version of Tensor '" << intermidiate_tensor_.name()
            << "' is [ " << tensor_version << " ]";
  }
}

}  // namespace egr
