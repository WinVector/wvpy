
import os
import pytest
from nbconvert.preprocessors.execute import CellExecutionError

from wvpy.jtools import render_as_html


# Was seeing:
# nbconvert/filters/ansi.py:60: DeprecationWarning: 'jinja2.escape' is deprecated and will be removed in Jinja 3.1. Import 'markupsafe.escape' instead.
@pytest.mark.filterwarnings("ignore:")
def test_jupyter_notebook_good():
    orig_wd = os.getcwd()
    source_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(source_dir)
    render_as_html(
        "example_good_notebook.ipynb"
    )
    os.remove("example_good_notebook.html")
    # want the raised issue if not present
    os.chdir(orig_wd)


# Was seeing:
# nbconvert/filters/ansi.py:60: DeprecationWarning: 'jinja2.escape' is deprecated and will be removed in Jinja 3.1. Import 'markupsafe.escape' instead.
@pytest.mark.filterwarnings("ignore:")
def test_jupyter_notebook_bad():
    orig_wd = os.getcwd()
    source_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(source_dir)
    with pytest.raises(CellExecutionError):
        render_as_html(
            "example_bad_notebook.ipynb"
        )
    os.chdir(orig_wd)

