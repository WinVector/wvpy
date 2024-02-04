import os
from wvpy.ptools import execute_py


def test_pysheet_param_1():
    source_dir = os.path.dirname(os.path.realpath(__file__))
    orig_wd = os.getcwd()
    os.chdir(source_dir)
    try:
        os.remove("example_parameterized_py_sheet.txt")
    except FileNotFoundError:
        pass
    execute_py("example_parameterized_py_sheet.py", sheet_vars={"x": 7})
    os.remove("example_parameterized_py_sheet.txt")
    # want the raised issue if not present
    os.chdir(orig_wd)
