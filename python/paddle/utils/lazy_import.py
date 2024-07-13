# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Lazy imports for heavy dependencies."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


__all__ = []


def try_import(module_name: str, err_msg: str | None = None) -> ModuleType:
    """Try importing a module, with an informative error message on failure."""
    install_name = module_name

    if module_name.find('.') > -1:
        install_name = module_name.split('.')[0]

    if module_name == 'cv2':
        install_name = 'opencv-python'

    try:
        mod = importlib.import_module(module_name)
        return mod
    except ImportError:
        if err_msg is None:
            err_msg = (
                f"Failed importing {module_name}. This likely means that some paddle modules "
                "require additional dependencies that have to be "
                f"manually installed (usually with `pip install {install_name}`). "
            )
        raise ImportError(err_msg)
