#! python
import unittest
import os
import tempfile
import shutil
from sampcd_processor import find_all
from sampcd_processor import check_indent
from sampcd_processor import sampcd_extract_and_run
from sampcd_processor import single_defcom_extract
from sampcd_processor import srccoms_extract

class Test_find_all(unittest.TestCase):
    # def test_srcstr_is_None(self):
    #    self.assertIsNone(find_all(None, 'hello world'))
    def test_find_none(self):
        self.assertEqual(0, len(find_all('hello', 'world')))
    def test_find_one(self):
        self.assertListEqual([0], find_all('hello', 'hello'))
    def test_find_two(self):
        self.assertListEqual([1, 15], find_all(' hello, world; hello paddle!', 'hello'))

class Test_check_indent(unittest.TestCase):
    def test_no_indent(self):
        self.assertEqual(0, check_indent('hello paddle'))
    def test_indent_4_spaces(self):
        self.assertEqual(4, check_indent('    hello paddle'))
    def test_indent_1_tab(self):
        self.assertEqual(4, check_indent("\thello paddle"))
    # def test_indent_mixed_spaces_and_tab(self):
    #     with self.assertRaises(Exception):
    #         check_indent("  \thello paddle")

class Test_sampcd_extract_and_run(unittest.TestCase):
    def setUp(self):
        if not os.path.exists('samplecode_temp/'):
            os.mkdir('samplecode_temp/')
    def test_run_a_defs_samplecode(self):
        comments = """
        Examples:
            .. code-block:: python
                print(1+1)
        """
        funcname = 'one_plus_one'
        self.assertTrue(sampcd_extract_and_run(comments, funcname))
    def test_run_a_def_no_code(self):
        comments = """
        placeholder
        """
        funcname = 'one_plus_one'
        self.assertFalse(sampcd_extract_and_run(comments, funcname))
    def test_run_a_def_raise_expection(self):
        comments = """
        placeholder
        Examples:
            .. code-block:: python
                print(1/0)
        """
        funcname = 'one_plus_one'
        self.assertFalse(sampcd_extract_and_run(comments, funcname))

class Test_single_defcom_extract(unittest.TestCase):
    def test_extract_from_func(self):
        defstr='''
        import os
        def foo():
            """
            foo is a function.
            """
            pass
        def bar():
            pass
        '''
        comm = single_defcom_extract(0, defstr.splitlines(True), is_class_begin=False)
        self.assertEqual("            foo is a function.\n", comm)
        pass
    def test_extract_from_func_with_no_docstring(self):
        defstr='''
        import os
        def bar():
            pass
        '''
        comm = single_defcom_extract(0, defstr.splitlines(True), is_class_begin=False)
        self.assertEqual('', comm)
        pass
    def test_extract_from_class(self):
        defstr='''
        import os
        class Foo():
            r"""
            foo is a class.
            second line.
            """
            pass
            def bar():
                pass
        def foo():
            pass
        '''
        comm = single_defcom_extract(0, defstr.splitlines(True), is_class_begin=True)
        rcomm = r"""            foo is a class.
            second line.
"""
        self.assertEqual(rcomm, comm)
        pass
    def test_extract_from_class_with_no_docstring(self):
        defstr='''
        import os
        class Foo():
            pass
            def bar():
                pass
        def foo():
            pass
        '''
        comm = single_defcom_extract(0, defstr.splitlines(True), is_class_begin=True)
        self.assertEqual('', comm)

class Test_srccoms_extract(unittest.TestCase):
    def setUp(self):
        self.tmpDir = tempfile.mkdtemp()
    def tearDown(self):
        shutil.rmtree(self.tmpDir)
    def test_from_ops_py(self):
        pass
    def test_from_not_ops_py(self):
        filecont = r'''
        __all__ = [
        'one_plus_one'
        ]

        def one_plus_one():
            """
            placeholder

            Examples:
            .. code-block:: python
                print(1+1)
            """
            return 1+1

        '''
        pyfilename = os.path.join(self.tmpDir, 'opo.py')
        with open(pyfilename, 'w') as pyfile:
            pyfile.write(filecont)
        with open(pyfilename, 'r') as pyfile:
            self.assertTrue(srccoms_extract(pyfile, []))
        pass
    def test_with_empty_wlist(self):
        pass
    def test_with_wlist(self):
        pass

# https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/layers/ops.py
# why? unabled to use the ast module. emmmmm

if __name__ == '__main__':
    unittest.main()
