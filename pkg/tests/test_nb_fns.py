
import os
import pytest
from nbconvert.preprocessors.execute import CellExecutionError

from wvpy.jtools import render_as_html, convert_py_code_to_notebook, convert_notebook_code_to_py


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



ex_txt = """
1 + 2

'''end code'''

6 - 7

''' begin text
*hello* 
world
'''  # end text


''' begin text
txt2
'''  # end text

(3
 + 4)
"""

@pytest.mark.filterwarnings("ignore:")
def test_jupyter_notebook_parameterized_good():
    orig_wd = os.getcwd()
    source_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(source_dir)
    render_as_html(
        "example_parameterized_notebook.ipynb",
        init_code='x = 2',
    )
    os.remove("example_parameterized_notebook.html")
    # want the raised issue if not present
    os.chdir(orig_wd)


def test_nb_convert():
    nb = convert_py_code_to_notebook(ex_txt)
    res_txt = convert_notebook_code_to_py(nb)
    # TODO: compare to orignal text
    
