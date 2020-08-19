#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

# gast is a generic AST to represent Python2 and Python3's Abstract Syntax Tree(AST).
# It provides a compatibility layer between the AST of various Python versions,
# as produced by ast.parse from the standard ast module.
# See details in https://github.com/serge-sans-paille/gast/
import gast

from paddle.fluid.dygraph.dygraph_to_static.assert_transformer import AssertTransformer
from paddle.fluid.dygraph.dygraph_to_static.basic_api_transformer import BasicApiTransformer
from paddle.fluid.dygraph.dygraph_to_static.break_continue_transformer import BreakContinueTransformer
from paddle.fluid.dygraph.dygraph_to_static.call_transformer import CallTransformer
from paddle.fluid.dygraph.dygraph_to_static.cast_transformer import CastTransformer
from paddle.fluid.dygraph.dygraph_to_static.ifelse_transformer import IfElseTransformer
from paddle.fluid.dygraph.dygraph_to_static.list_transformer import ListTransformer
from paddle.fluid.dygraph.dygraph_to_static.logical_transformer import LogicalTransformer
from paddle.fluid.dygraph.dygraph_to_static.loop_transformer import LoopTransformer
from paddle.fluid.dygraph.dygraph_to_static.print_transformer import PrintTransformer
from paddle.fluid.dygraph.dygraph_to_static.return_transformer import ReturnTransformer
from paddle.fluid.dygraph.dygraph_to_static.tensor_shape_transformer import TensorShapeTransformer

from paddle.fluid.dygraph.dygraph_to_static.static_analysis import StaticAnalysisVisitor
from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_source_code
from paddle.fluid.dygraph.dygraph_to_static.utils import get_attribute_full_name
from paddle.fluid.dygraph.dygraph_to_static.logging_utils import TranslatorLogger

__all__ = ['DygraphToStaticAst']

DECORATOR_NAMES = ['declarative', 'to_static', 'dygraph_to_static_func']


class DygraphToStaticAst(gast.NodeTransformer):
    """
    Main class to transform Dygraph to Static Graph
    """

    def get_static_ast(self, root):
        # save root for some analysis may need global AST
        self.root = root
        self.static_analysis_visitor = StaticAnalysisVisitor(root)
        self.static_analysis_root = self.static_analysis_visitor.get_node_wrapper_root(
        )
        self.decorate_func_name = None
        self.transfer_from_node_type(self.static_analysis_root)
        return self.static_analysis_root

    def transfer_from_node_type(self, node_wrapper):
        translator_logger = TranslatorLogger()
        translator_logger.log(
            1, "   Source code: \n{}".format(ast_to_source_code(self.root)))
        # Generic transformation
        self.visit(node_wrapper.node)

        # Transform basic api of dygraph to static graph and get feed_name_to_arg_name
        BasicApiTransformer(node_wrapper).transform()
        translator_logger.log_transformed_code("BasicApiTransformer", self.root)

        # Transform Tensor.shape into fluid.layers.shape(Tensor)
        TensorShapeTransformer(node_wrapper).transform()
        translator_logger.log_transformed_code("TensorShapeTransformer",
                                               self.root)

        # Transform list used in control flow
        ListTransformer(node_wrapper).transform()
        translator_logger.log_transformed_code("ListTransformer", self.root)

        # Transform break/continue in loops
        BreakContinueTransformer(node_wrapper).transform()
        translator_logger.log_transformed_code("BreakContinueTransformer",
                                               self.root)

        # Transform return in functions
        ReturnTransformer(node_wrapper).transform()
        translator_logger.log_transformed_code("ReturnTransformer", self.root)

        # Transform logical and/or/not
        LogicalTransformer(node_wrapper).transform()
        translator_logger.log_transformed_code("LogicalTransformer", self.root)

        # Transform for loop and while loop
        LoopTransformer(node_wrapper).transform()
        translator_logger.log_transformed_code("LoopTransformer", self.root)

        # Transform all if/else statement of Dygraph into Static Graph.
        IfElseTransformer(node_wrapper).transform()
        translator_logger.log_transformed_code("IfElseTransformer", self.root)

        # Transform python assert statement
        AssertTransformer(node_wrapper).transform()
        translator_logger.log_transformed_code("AssertTransformer", self.root)

        # Transform all python print statement
        PrintTransformer(node_wrapper).transform()
        translator_logger.log_transformed_code("PrintTransformer", self.root)

        # Transform call recursively
        CallTransformer(node_wrapper).transform()
        translator_logger.log_transformed_code("CallTransformer", self.root)

        # Transform python type casting statement
        CastTransformer(node_wrapper).transform()
        translator_logger.log_transformed_code("CastTransformer", self.root)

    def visit_FunctionDef(self, node):
        if self.decorate_func_name is None:
            self.decorate_func_name = node.name

        self.generic_visit(node)
        # Remove the decorated name of dygraph_to_static
        if hasattr(node, 'decorator_list'):
            decorator_list = []
            for d in node.decorator_list:
                if isinstance(d, gast.Name) and d.id not in DECORATOR_NAMES:
                    raise NotImplementedError(
                        "ProgramTranslator hasn't implemented multiple decorators. Please remove "
                        + d.id + " in " + self.decorate_func_name)
                if isinstance(d, gast.Attribute):
                    full_attribute_name = get_attribute_full_name(d)
                    has_translate_decorator = False
                    for deco in DECORATOR_NAMES:
                        if deco in full_attribute_name:
                            has_translate_decorator = True
                            break
                    if not has_translate_decorator:
                        raise NotImplementedError(
                            "ProgramTranslator hasn't implemented multiple decorators. Please remove "
                            + full_attribute_name + " in " +
                            self.decorate_func_name)
            node.decorator_list = decorator_list
        return node

    def get_module_name(self):
        """
        Return the main function name which will be used as module name
        in ast_to_func.
        """
        # Should consider BaseAPITransformer which add new module name in Yamei's PR.
        assert self.decorate_func_name, "decorate_func_name shall not be None."
        return self.decorate_func_name
