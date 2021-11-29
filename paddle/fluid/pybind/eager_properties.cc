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
// disable numpy compile error
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL Paddle_PyArray_API_P
#define INIT_NUMPY_ARRAY_CPP

#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

#include <Python.h>

#include <string>
#include <vector>

#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/include/core.h"
#pragma GCC diagnostic ignored "-Wconversion-null"  // for import_array();
#pragma GCC diagnostic ignored "-Wunused-variable"  // for numpy_initialized_p
#pragma GCC diagnostic ignored \
    "-Wwrite-strings"  // for {nullptr, nullptr, nullptr, nullptr, nullptr}};

namespace paddle {
namespace pybind {

int init_numpy_p() {
  import_array();
  return 0;
}
static const int numpy_initialized_p = init_numpy_p();

extern PyTypeObject* p_eager_tensor_type;

PyObject* eager_tensor_properties_get_name(EagerTensorObject* self,
                                           void* closure) {
  return ToPyObject(self->eagertensor.name());
}

int eager_tensor_properties_set_name(EagerTensorObject* self, PyObject* value,
                                     void* closure) {
  self->eagertensor.set_name(CastPyArg2AttrString(value, 0));
  return 0;
}

PyObject* eager_tensor_properties_get_stop_gradient(EagerTensorObject* self,
                                                    void* closure) {
  auto meta = egr::EagerUtils::unsafe_autograd_meta(self->eagertensor);
  return ToPyObject(meta->StopGradient());
}

PyObject* eager_tensor_properties_get_grad(EagerTensorObject* self,
                                           void* closure) {
  auto meta = egr::EagerUtils::unsafe_autograd_meta(self->eagertensor);
  return ToPyObject(meta->Grad());
}

int eager_tensor_properties_set_stop_gradient(EagerTensorObject* self,
                                              PyObject* value, void* closure) {
  auto meta = egr::EagerUtils::unsafe_autograd_meta(self->eagertensor);
  meta->SetStopGradient(CastPyArg2AttrBoolean(value, 0));
  return 0;
}

PyObject* eager_tensor_properties_get_persistable(EagerTensorObject* self,
                                                  void* closure) {
  auto meta = egr::EagerUtils::unsafe_autograd_meta(self->eagertensor);
  return ToPyObject(meta->Persistable());
}

int eager_tensor_properties_set_persistable(EagerTensorObject* self,
                                            PyObject* value, void* closure) {
  auto meta = egr::EagerUtils::unsafe_autograd_meta(self->eagertensor);
  meta->SetPersistable(CastPyArg2AttrBoolean(value, 0));
  return 0;
}

PyObject* eager_tensor_properties_get_shape(EagerTensorObject* self,
                                            void* closure) {
  auto ddim = self->eagertensor.shape();
  std::vector<int64_t> value;
  size_t rank = static_cast<size_t>(ddim.size());
  value.resize(rank);
  for (size_t i = 0; i < rank; i++) {
    value[i] = ddim[i];
  }

  return ToPyObject(value);
}

PyObject* eager_tensor_properties_get_place(EagerTensorObject* self,
                                            void* closure) {
  return ToPyObject(self->eagertensor.place());
}

PyObject* eager_tensor_properties_get_place_str(EagerTensorObject* self,
                                                void* closure) {
  std::stringstream ostr;
  ostr << self->eagertensor.place();
  return ToPyObject(ostr.str());
}

PyObject* eager_tensor_properties_get_dtype(EagerTensorObject* self,
                                            void* closure) {
  return ToPyObject(pten::DataType2String(self->eagertensor.type()));
}

struct PyGetSetDef variable_properties[] = {
    {"grad", (getter)eager_tensor_properties_get_grad, nullptr, nullptr,
     nullptr},
    {"name", (getter)eager_tensor_properties_get_name,
     (setter)eager_tensor_properties_set_name, nullptr, nullptr},
    {"stop_gradient", (getter)eager_tensor_properties_get_stop_gradient,
     (setter)eager_tensor_properties_set_stop_gradient, nullptr, nullptr},
    {"persistable", (getter)eager_tensor_properties_get_persistable,
     (setter)eager_tensor_properties_set_persistable, nullptr, nullptr},
    {"shape", (getter)eager_tensor_properties_get_shape, nullptr, nullptr,
     nullptr},
    // {"is_leaf", (getter)eager_tensor_properties_get_is_leaf, nullptr,
    // nullptr,
    //  nullptr},
    {"place", (getter)eager_tensor_properties_get_place, nullptr, nullptr,
     nullptr},
    {"_place_str", (getter)eager_tensor_properties_get_place_str, nullptr,
     nullptr, nullptr},
    {"dtype", (getter)eager_tensor_properties_get_dtype, nullptr, nullptr,
     nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

}  // namespace pybind
}  // namespace paddle
