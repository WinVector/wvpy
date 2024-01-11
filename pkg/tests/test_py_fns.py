
import os
from wvpy.ptools import execute_py


def test_execute_py_1():
    source_dir = os.path.dirname(os.path.realpath(__file__))
    orig_wd = os.getcwd()
    os.chdir(source_dir)
    try:
        os.remove("example_py_sheet.txt")
    except FileNotFoundError:
        pass
    execute_py(
        "example_py_sheet.py"
    )
    os.remove("example_py_sheet.txt")
    # want the raised issue if not present
    os.chdir(orig_wd)


