import os
from wvpy.render_workbook import render_workbook


def test_pysheet_1():
    source_dir = os.path.dirname(os.path.realpath(__file__))
    orig_wd = os.getcwd()
    os.chdir(source_dir)
    try:
        os.remove("example_good_notebook.html")
    except FileNotFoundError:
        pass
    render_workbook(["example_good_notebook.ipynb"])
    os.remove("example_good_notebook.html")
    # want the raised issue if not present
    os.chdir(orig_wd)
