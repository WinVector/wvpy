import os
from wvpy.pysheet import pysheet


def test_pysheet_1():
    source_dir = os.path.dirname(os.path.realpath(__file__))
    orig_wd = os.getcwd()
    os.chdir(source_dir)
    try:
        os.remove("example_good_notebook.py")
    except FileNotFoundError:
        pass
    pysheet(["example_good_notebook.ipynb"])
    os.remove("example_good_notebook.py")
    # want the raised issue if not present
    os.chdir(orig_wd)
