# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INDENT_SIZE = 4
INDENT = " " * INDENT_SIZE

# TODO(SigureMo): add comments...


@dataclass
class Location:
    lineno: int
    col_offset: int


class TensorGen:
    _future_features: list[str]
    _attributes: dict[str, str]
    _methods: list[str]
    _aliases: dict[str, str]

    def __init__(self):
        self._future_features: list[str] = []
        self._attributes: dict[str, str] = {}
        self._methods: list[str] = []
        self._aliases: dict[str, str] = {}
        self.add_future("annotations")

    def add_future(self, feature: str):
        self._future_features.append(feature)

    def add_attribute(self, name: str, type: str):
        self._attributes[name] = type

    def add_method(self, func: str):
        self._methods.append(func)

    def add_alias(self, alias: str, target: str):
        self._aliases[alias] = target

    @property
    def future_imports(self):
        futures = ", ".join(self._future_features)
        return f"from __future__ import {futures}"

    @property
    def tensor_spec(self) -> str:
        return """
class Tensor:
"""

    @property
    def tensor_attributes(self) -> str:
        attributes_code = ""
        for name, type_ in self._attributes.items():
            attributes_code += f"{INDENT}{name}: {type_}\n"
        return attributes_code

    @property
    def tensor_methods(self) -> str:
        method_code = ""
        for method in self._methods:
            method_code += method + "\n"
        return method_code

    @property
    def tensor_aliases(self) -> str:
        aliases_code = ""
        for alias, target in self._aliases.items():
            aliases_code += f"{INDENT}{alias} = {target}\n"
        return aliases_code

    def codegen(self) -> str:
        code = f"""\
# This file is auto-generated by tools/gen_tensor_proxy.py

{self.future_imports}

{self.tensor_spec}

{self.tensor_attributes}

{self.tensor_methods}

{self.tensor_aliases}
"""
        return code


def eval_ast(node: ast.AST) -> Any:
    if isinstance(node, ast.List):
        return [eval_ast(elt) for elt in node.elts]
    elif isinstance(node, ast.Constant):
        return node.value
    else:
        raise TypeError(f"Cannot eval ast node: {node}")


def crop_string(text: str, start: Location, end: Location) -> str:
    lines_length = [len(line) + 1 for line in text.splitlines()]
    start_index = sum(lines_length[: start.lineno - 1]) + start.col_offset
    end_index = sum(lines_length[: end.lineno - 1]) + end.col_offset
    return text[start_index:end_index]


def _func_spec_to_tensor_method(func_code: str) -> str:
    method_code = ""

    # Replace first argument with self
    regex_func_spec = re.compile(
        r"^def (?P<method_name>[_a-zA-Z0-9]+)\((?P<arg0>[^,)]+(:.+)?)(?P<rest_args>.+)\)",
        re.DOTALL,
    )
    matched = regex_func_spec.search(func_code)
    if matched is None:
        raise ValueError(f"Cannot parse function spec: {func_code}")
    func_code = regex_func_spec.sub(
        r"def \g<method_name>(self\g<rest_args>)", func_code
    )

    # Iterate every line, insert the indent and remove document of the first argument
    is_first_arg = False
    first_arg_offset = 0

    for line in func_code.splitlines():
        current_line_offset = len(line) - len(line.lstrip())
        # Remove the first argument (self in Tensor method) from docstring
        if is_first_arg:
            if current_line_offset <= first_arg_offset:
                is_first_arg = False
            if not first_arg_offset:
                first_arg_offset = current_line_offset
            if is_first_arg:
                continue
        method_code += f"{INDENT}{line}\n" if line else "\n"
        if line.lstrip().startswith("Args:"):
            is_first_arg = True

    return method_code


def get_tensor_method_func(tensor_funcs_dir: Path) -> list[str]:
    tensor_init_path = tensor_funcs_dir / "__init__.py"
    with open(tensor_init_path, "r") as f:
        tensor_init_ast = ast.parse(f.read())
    for node in ast.walk(tensor_init_ast):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "tensor_method_func"
        ):
            tensor_method_func_ast = node.value
            break
    else:
        raise ValueError(
            "Cannot find tensor_method_func in python/paddle/tensor__init__.py"
        )
    return eval_ast(tensor_method_func_ast)


def get_tensor_monkey_patched_methods(
    tensor_funcs_dir: Path, tensor_method_func: list[str]
) -> list[str]:
    methods: list[str] = []
    for tensor_funcs_path in tensor_funcs_dir.iterdir():
        if (
            tensor_funcs_path.is_dir()
            or tensor_funcs_path.name == "__init__.py"
            or tensor_funcs_path.name == "tensor_proxy.py"
        ):
            continue
        with open(tensor_funcs_path, "r") as f:
            source = f.read()
            tensor_funcs_ast = ast.parse(source)
        for node in ast.walk(tensor_funcs_ast):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name in tensor_method_func
            ):
                start_location = Location(node.lineno, node.col_offset)
                end_location = Location(
                    node.body[0].lineno, node.body[0].col_offset
                )
                if (
                    node.body
                    and isinstance(
                        node.body[0], ast.Expr
                    )  # docstring is the first statement
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    end_location = Location(
                        node.body[0].end_lineno, node.body[0].end_col_offset  # type: ignore
                    )

                func_spec_with_docstring = crop_string(
                    source, start_location, end_location
                )
                func_spec_with_docstring_and_ellipsis = (
                    f"{func_spec_with_docstring}\n{INDENT}...\n"
                )
                method_code = _func_spec_to_tensor_method(
                    func_spec_with_docstring_and_ellipsis
                )

                methods.append(method_code)
    return methods


def get_alias() -> dict[str, str]:
    # TODO: Add the reference from the source code
    return {
        "reverse": "flip",
        "mod": "remainder",
        "floor_mod": "remainder",
    }


def get_attributes() -> dict[str, str]:
    # TODO: add more attributes
    return {
        "shape": "list[int]",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tensor-funcs-dir", type=str, default="python/paddle/tensor/"
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="python/paddle/tensor/tensor_proxy.py",
    )

    args = parser.parse_args()

    # Get methods of Tensor
    tensor_funcs_dir = Path(args.tensor_funcs_dir)
    tensor_method_func = get_tensor_method_func(tensor_funcs_dir)
    tensor_monkey_patched_methods = get_tensor_monkey_patched_methods(
        tensor_funcs_dir, tensor_method_func
    )
    tensor_aliases = get_alias()
    tensor_attributes = get_attributes()

    # Generate the proxy Tensor class
    tensor_gen = TensorGen()
    for attr, type in tensor_attributes.items():
        tensor_gen.add_attribute(attr, type)
    for method in tensor_monkey_patched_methods:
        tensor_gen.add_method(method)
    for aliases, target in tensor_aliases.items():
        tensor_gen.add_alias(aliases, target)

    # Write to target file
    with open(args.output_file, "w") as f:
        f.write(tensor_gen.codegen())


if __name__ == "__main__":
    main()
