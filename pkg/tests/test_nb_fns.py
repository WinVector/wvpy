
import os
import pytest
import warnings
import multiprocessing

# importing nbconvert components such as nbconvert.preprocessors.execute
# while in pytest causes the following warning:
#
# jupyter_client/connect.py:27: DeprecationWarning: Jupyter is migrating its paths to use standard platformdirs
#   given by the platformdirs library.  To remove this warning and
#   see the appropriate new directories, set the environment variable
#   `JUPYTER_PLATFORM_DIRS=1` and then run `jupyter --paths`.
#   The use of platformdirs will be the default in `jupyter_core` v6
#     from jupyter_core.paths import jupyter_data_dir
# -- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
#
# likely this is from some environment isolation set up by pytest
# the same code does not warn when executing directly
#
# so we are going to trigger the warning on first import, so it is caught
# and does not reoccur when we import wvpy.jtools components
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from nbconvert.preprocessors.execute import CellExecutionError  # even importing this causes warning

from wvpy.jtools import render_as_html, convert_py_code_to_notebook, convert_notebook_code_to_py, JTask, job_fn, run_pool


# confirm we have not killed all warnings
def test_still_warns_even_after_monkeying():
    with pytest.warns(DeprecationWarning):
        warnings.warn(DeprecationWarning("test warning"))


def test_jupyter_notebook_good():
    source_dir = os.path.dirname(os.path.realpath(__file__))
    orig_wd = os.getcwd()
    os.chdir(source_dir)
    try:
        os.remove("example_good_notebook.html")
    except FileNotFoundError:
        pass
    render_as_html(
        "example_good_notebook.ipynb"
    )
    os.remove("example_good_notebook.html")
    # want the raised issue if not present
    os.chdir(orig_wd)


def test_jupyter_notebook_sheet_vars_good():
    source_dir = os.path.dirname(os.path.realpath(__file__))
    orig_wd = os.getcwd()
    os.chdir(source_dir)
    try:
        os.remove("example_sheet_vars_notebook.html")
    except FileNotFoundError:
        pass
    render_as_html(
        "example_sheet_vars_notebook.ipynb",
        sheet_vars={'x': 7},
        init_code='y = 3',
    )
    os.remove("example_sheet_vars_notebook.html")
    # want the raised issue if not present
    os.chdir(orig_wd)


def test_jupyter_notebook_sheet_vars_bad():
    source_dir = os.path.dirname(os.path.realpath(__file__))
    orig_wd = os.getcwd()
    os.chdir(source_dir)
    try:
        os.remove("example_sheet_vars_notebook.html")
    except FileNotFoundError:
        pass
    with pytest.raises(CellExecutionError):
        render_as_html(
            "example_sheet_vars_notebook.ipynb",
            sheet_vars={'x': 4},  # wrong value!
            init_code='y = 3',
        )
    os.remove("example_sheet_vars_notebook.html")
    # want the raised issue if not present
    os.chdir(orig_wd)


def test_jupyter_notebook_bad():
    source_dir = os.path.dirname(os.path.realpath(__file__))
    orig_wd = os.getcwd()
    os.chdir(source_dir)
    try:
        os.remove("example_bad_notebook.html")
    except FileNotFoundError:
        pass
    with pytest.raises(CellExecutionError):
        render_as_html(
            "example_bad_notebook.ipynb"
        )
    os.remove("example_bad_notebook.html")
    os.chdir(orig_wd)


def test_pool_completes_bad():
    source_dir = os.path.dirname(os.path.realpath(__file__))
    orig_wd = os.getcwd()
    os.chdir(source_dir)
    for nm in ["example_good_notebook.html", "example_parameterized_notebook.html"]:
        try:
            os.remove(nm)
        except FileNotFoundError:
            pass
    tasks = [
        JTask("example_good_notebook.ipynb"),
        JTask("example_parameterized_notebook.ipynb"),
    ]
    run_pool(
        tasks,
        njobs=2,
        verbose=False,
        stop_on_error=False,
    )
    for nm in ["example_good_notebook.html", "example_parameterized_notebook.html"]:
        os.remove(nm)
    os.chdir(orig_wd)


def test_pool_stops_bad():
    source_dir = os.path.dirname(os.path.realpath(__file__))
    orig_wd = os.getcwd()
    os.chdir(source_dir)
    for nm in ["example_good_notebook.html", "example_parameterized_notebook.html"]:
        try:
            os.remove(nm)
        except FileNotFoundError:
            pass
    tasks = [
        JTask("example_good_notebook.ipynb"),
        JTask("example_parameterized_notebook.ipynb"),
    ]
    with pytest.raises(CellExecutionError):
        run_pool(
            tasks,
            njobs=2,
            verbose=False,
            stop_on_error=True,
        )
    for nm in ["example_good_notebook.html", "example_parameterized_notebook.html"]:
        os.remove(nm)
    os.chdir(orig_wd)


def test_jupyter_notebook_parameterized_good():
    source_dir = os.path.dirname(os.path.realpath(__file__))
    orig_wd = os.getcwd()
    os.chdir(source_dir)
    try:
        os.remove("example_parameterized_notebook.html")
    except FileNotFoundError:
        pass
    render_as_html(
        "example_parameterized_notebook.ipynb",
        init_code='x = 2',
    )
    os.remove("example_parameterized_notebook.html")
    # want the raised issue if not present
    os.chdir(orig_wd)


def test_jupyter_notebook_parameterized_bad():
    source_dir = os.path.dirname(os.path.realpath(__file__))
    orig_wd = os.getcwd()
    os.chdir(source_dir)
    try:
        os.remove("example_parameterized_notebook.html")
    except FileNotFoundError:
        pass
    with pytest.raises(CellExecutionError):
        render_as_html(
            "example_parameterized_notebook.ipynb",
        )
    os.remove("example_parameterized_notebook.html")
    # want the raised issue if not present
    os.chdir(orig_wd)


def test_jtask_param_good():
    source_dir = os.path.dirname(os.path.realpath(__file__))
    orig_wd = os.getcwd()
    os.chdir(source_dir)
    task = JTask(
        "example_parameterized_notebook.ipynb",
        init_code='x = 2',
        output_suffix="_z",
    )
    task_str = str(task)
    assert isinstance(task_str, str)
    back = eval(task_str)
    assert task == back
    try:
        os.remove("example_parameterized_notebook_z.html")
    except FileNotFoundError:
        pass
    with multiprocessing.Pool(2) as p:
        p.map(job_fn, [task])
    os.remove("example_parameterized_notebook_z.html")
    # want the raised issue if not present
    os.chdir(orig_wd)


def test_jtask_param_bad():
    source_dir = os.path.dirname(os.path.realpath(__file__))
    orig_wd = os.getcwd()
    os.chdir(source_dir)
    try:
        os.remove("example_parameterized_notebook.html")
    except FileNotFoundError:
        pass
    task = JTask(
        "example_parameterized_notebook.ipynb",
        init_code='x = 1',
    )
    with pytest.raises(CellExecutionError):
        with multiprocessing.Pool(2) as p:
            p.map(job_fn, [task])
    os.remove("example_parameterized_notebook.html")
    # want the raised issue if not present
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


def test_nb_convert():
    nb = convert_py_code_to_notebook(ex_txt)
    res_txt = convert_notebook_code_to_py(nb)
    assert isinstance(res_txt, str)
    # TODO: compare to original text


def test_JTask_basics():
    t1 = JTask("example_good_notebook.ipynb", strict=False)
    s1 = str(t1)
    assert isinstance(s1, str)
    r1 = t1.__repr__()
    assert isinstance(r1, str)
    t2 = JTask("example_good_notebook.ipynb", strict=False, sheet_vars={'x': 7})
    s2 = str(t2)
    assert isinstance(s2, str)
    r2 = t2.__repr__()
    assert isinstance(r2, str)
