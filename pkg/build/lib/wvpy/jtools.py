"""Jupyter tools"""

import re
import datetime
import os
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from multiprocessing import Pool
import sys
import pickle
import tempfile
import traceback

from typing import Any, Dict, Iterable, List, Optional
from contextlib import contextmanager

from wvpy.util import escape_ansi
from wvpy.ptools import execute_py


have_black = False
try:
    import black

    have_black = True
except ModuleNotFoundError:
    pass

nbf_v = nbformat.v4

# def _normalize(nb):
#    # try and work around version mismatches
#    return nbformat.validator.normalize(nb, version=4, version_minor=5)[1]


# noinspection PyBroadException
def pretty_format_python(python_txt: str, *, black_mode=None) -> str:
    """
    Format Python code, using black.

    :param python_txt: Python code
    :param black_mode: options for black
    :return: formatted Python code
    """
    assert have_black
    assert isinstance(python_txt, str)
    formatted_python = python_txt.strip("\n") + "\n"
    if len(formatted_python.strip()) > 0:
        if black_mode is None:
            black_mode = black.FileMode()
        try:
            formatted_python = black.format_str(formatted_python, mode=black_mode)
            formatted_python = formatted_python.strip("\n") + "\n"
        except Exception:
            pass
    return formatted_python


def convert_py_code_to_notebook(
    text: str, *, use_black: bool = False
) -> nbformat.notebooknode.NotebookNode:
    """
    Convert python text to a notebook.
    "''' begin text" ends any open blocks, and starts a new markdown block (triple double quotes also allowed)
    "''' # end text" ends text, and starts a new code block (triple double quotes also allowed)
    "'''end code'''" ends code blocks, and starts a new code block (triple double quotes also allowed)

    :param text: Python text to convert.
    :param use_black: if True use black to re-format Python code
    :return: a notebook
    """
    # https://stackoverflow.com/a/23729611/6901725
    # https://nbviewer.org/gist/fperez/9716279
    assert isinstance(text, str)
    assert isinstance(use_black, bool)
    lines = text.splitlines()
    begin_text_regexp = re.compile(r"^\s*r?((''')|(\"\"\"))\s*begin\s+text\s*$")
    end_text_regexp = re.compile(r"^\s*r?((''')|(\"\"\"))\s*#\s*end\s+text\s*$")
    end_code_regexp = re.compile(
        r"(^\s*r?'''\s*end\s+code\s*'''\s*$)|(^\s*r?\"\"\"\s*end\s+code\s*\"\"\"\s*$)"
    )
    # run a little code collecting state machine
    cells = []
    collecting_python = []
    collecting_text = None
    lines.append(None)  # append an ending sentinel
    # scan input
    for line in lines:
        if line is None:
            is_end = True
            text_start = False
            code_start = False
            code_end = False
        else:
            is_end = False
            text_start = begin_text_regexp.match(line)
            code_start = end_text_regexp.match(line)
            code_end = end_code_regexp.match(line)
        if is_end or text_start or code_start or code_end:
            if (collecting_python is not None) and (len(collecting_python) > 0):
                python_block = ("\n".join(collecting_python)).strip("\n") + "\n"
                if len(python_block.strip()) > 0:
                    if use_black and have_black:
                        python_block = pretty_format_python(python_block)
                    cells.append(nbf_v.new_code_cell(python_block))
            if (collecting_text is not None) and (len(collecting_text) > 0):
                txt_block = ("\n".join(collecting_text)).strip("\n") + "\n"
                if len(txt_block.strip()) > 0:
                    cells.append(nbf_v.new_markdown_cell(txt_block))
            collecting_python = None
            collecting_text = None
            if not is_end:
                if text_start:
                    collecting_text = []
                else:
                    collecting_python = []
        else:
            if collecting_python is not None:
                collecting_python.append(line)
            if collecting_text is not None:
                collecting_text.append(line)
    for i in range(len(cells)):
        cells[i]["id"] = f"cell{i}"
    nb = nbf_v.new_notebook(cells=cells)
    # nb = _normalize(nb)
    return nb


def prepend_code_cell_to_notebook(
    nb: nbformat.notebooknode.NotebookNode,
    *,
    code_text: str,
) -> nbformat.notebooknode.NotebookNode:
    """
    Prepend a code cell to a Jupyter notebook.

    :param nb: Jupyter notebook to alter
    :param code_text: Python source code to add
    :return: new notebook
    """
    header_cell = nbf_v.new_code_cell(code_text)
    # set cell ids to avoid:
    # "MissingIDFieldWarning: Code cell is missing an id field, this will become a hard error in future nbformat versions."
    header_cell["id"] = "wvpy_header_cell"
    orig_cells = [c.copy() for c in nb.cells]
    for i in range(len(orig_cells)):
        orig_cells[i]["id"] = f"cell{i}"
    cells = [header_cell] + orig_cells
    nb_out = nbf_v.new_notebook(cells=cells)
    # nb_out = _normalize(nb_out)
    return nb_out


def convert_py_file_to_notebook(
    py_file: str,
    *,
    ipynb_file: str,
    use_black: bool = False,
) -> None:
    """
    Convert python text to a notebook.
    "''' begin text" ends any open blocks, and starts a new markdown block (triple double quotes also allowed)
    "''' # end text" ends text, and starts a new code block (triple double quotes also allowed)
    "'''end code'''" ends code blocks, and starts a new code block (triple double quotes also allowed)

    :param py_file: Path to python source file.
    :param ipynb_file: Path to notebook result file.
    :param use_black: if True use black to re-format Python code
    :return: nothing
    """
    assert isinstance(py_file, str)
    assert isinstance(ipynb_file, str)
    assert isinstance(use_black, bool)
    assert py_file != ipynb_file  # prevent clobber
    with open(py_file, "r", encoding="utf-8") as f:
        text = f.read()
    nb = convert_py_code_to_notebook(text, use_black=use_black)
    with open(ipynb_file, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


def convert_notebook_code_to_py(
    nb: nbformat.notebooknode.NotebookNode,
    *,
    use_black: bool = False,
) -> str:
    """
    Convert ipython notebook inputs to a py code.
    "''' begin text" ends any open blocks, and starts a new markdown block (triple double quotes also allowed)
    "''' # end text" ends text, and starts a new code block (triple double quotes also allowed)
    "'''end code'''" ends code blocks, and starts a new code block (triple double quotes also allowed)

    :param nb: notebook
    :param use_black: if True use black to re-format Python code
    :return: Python source code
    """
    assert isinstance(use_black, bool)
    res = []
    code_needs_end = False
    for cell in nb.cells:
        if len(cell.source.strip()) > 0:
            if cell.cell_type == "code":
                if code_needs_end:
                    res.append('\n"""end code"""\n')
                py_text = cell.source.strip("\n") + "\n"
                if use_black and have_black:
                    py_text = pretty_format_python(py_text)
                res.append(py_text)
                code_needs_end = True
            else:
                res.append('\n""" begin text')
                res.append(cell.source.strip("\n"))
                res.append('"""  # end text\n')
                code_needs_end = False
    res_text = "\n" + ("\n".join(res)).strip("\n") + "\n\n"
    return res_text


def convert_notebook_file_to_py(
    ipynb_file: str,
    *,
    py_file: str,
    use_black: bool = False,
) -> None:
    """
    Convert ipython notebook inputs to a py file.
    "''' begin text" ends any open blocks, and starts a new markdown block (triple double quotes also allowed)
    "''' # end text" ends text, and starts a new code block (triple double quotes also allowed)
    "'''end code'''" ends code blocks, and starts a new code block (triple double quotes also allowed)

    :param ipynb_file: Path to notebook input file.
    :param py_file: Path to python result file.
    :param use_black: if True use black to re-format Python code
    :return: nothing
    """
    assert isinstance(py_file, str)
    assert isinstance(ipynb_file, str)
    assert isinstance(use_black, bool)
    assert py_file != ipynb_file  # prevent clobber
    with open(ipynb_file, "rb") as f:
        nb = nbformat.read(f, as_version=4)
    py_source = convert_notebook_code_to_py(nb, use_black=use_black)
    with open(py_file, "w", encoding="utf-8") as f:
        f.write(py_source)


class OurExecutor(ExecutePreprocessor):
    """Catch exception in notebook processing"""

    def __init__(self, **kw):
        """Initialize the preprocessor."""
        ExecutePreprocessor.__init__(self, **kw)
        self.caught_exception = None

    def preprocess_cell(self, cell, resources, index):
        """
        Override if you want to apply some preprocessing to each cell.
        Must return modified cell and resource dictionary.

        Parameters
        ----------
        cell : NotebookNode cell
            Notebook cell being processed
        resources : dictionary
            Additional resources used in the conversion process.  Allows
            preprocessors to pass variables into the Jinja engine.
        index : int
            Index of the cell being processed
        """
        if self.caught_exception is None:
            try:
                return ExecutePreprocessor.preprocess_cell(self, cell, resources, index)
            except Exception as ex:
                self.caught_exception = ex
        return cell, self.resources


# https://nbconvert.readthedocs.io/en/latest/execute_api.html
# https://nbconvert.readthedocs.io/en/latest/nbconvert_library.html
# HTML element we are trying to delete:
#   <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[5]:</div>
def render_as_html(
    notebook_file_name: str,
    *,
    output_suffix: Optional[str] = None,
    timeout: int = 60000,
    kernel_name: Optional[str] = None,
    verbose: bool = True,
    sheet_vars=None,
    init_code: Optional[str] = None,
    exclude_input: bool = False,
    prompt_strip_regexp: Optional[
        str
    ] = r'<\s*div\s+class\s*=\s*"jp-OutputPrompt[^<>]*>[^<>]*Out[^<>]*<\s*/div\s*>',
) -> None:
    """
    Render a Jupyter notebook in the current directory as HTML.
    Exceptions raised in the rendering notebook are allowed to pass trough.

    :param notebook_file_name: name of source file, must end with .ipynb or .py (or type gotten from file system)
    :param output_suffix: optional name to add to result name
    :param timeout: Maximum time in seconds each notebook cell is allowed to run.
                    passed to nbconvert.preprocessors.ExecutePreprocessor.
    :param kernel_name: Jupyter kernel to use. passed to nbconvert.preprocessors.ExecutePreprocessor.
    :param verbose logical, if True print while running
    :param sheet_vars: if not None value is de-serialized as a variable named "sheet_vars"
    :param init_code: Python init code for first cell
    :param exclude_input: if True, exclude input cells
    :param prompt_strip_regexp: regexp to strip prompts, only used if exclude_input is True
    :return: None
    """
    assert isinstance(notebook_file_name, str)
    # deal with no suffix case
    if (not notebook_file_name.endswith(".ipynb")) and (
        not notebook_file_name.endswith(".py")
    ):
        py_name = notebook_file_name + ".py"
        py_exists = os.path.exists(py_name)
        ipynb_name = notebook_file_name + ".ipynb"
        ipynb_exists = os.path.exists(ipynb_name)
        if (py_exists + ipynb_exists) != 1:
            raise ValueError(
                "{ipynb_exists}: if file suffix is not specified then exactly one of .py or .ipynb file must exist"
            )
        if ipynb_exists:
            notebook_file_name = notebook_file_name + ".ipynb"
        else:
            notebook_file_name = notebook_file_name + ".py"
    # get the input
    assert os.path.exists(notebook_file_name)
    if notebook_file_name.endswith(".ipynb"):
        suffix = ".ipynb"
        with open(notebook_file_name, "rb") as f:
            nb = nbformat.read(f, as_version=4)
    elif notebook_file_name.endswith(".py"):
        suffix = ".py"
        with open(notebook_file_name, "r", encoding="utf-8") as f:
            text = f.read()
        nb = convert_py_code_to_notebook(text)
    else:
        raise ValueError("{ipynb_exists}: file must end with .py or .ipynb")
    tmp_path = None
    # do the conversion
    if sheet_vars is not None:
        with tempfile.NamedTemporaryFile(delete=False) as ntf:
            tmp_path = ntf.name
            pickle.dump(sheet_vars, file=ntf)
        if (init_code is None) or (len(init_code) <= 0):
            init_code = ""
        pickle_code = f"""
import pickle
with open({tmp_path.__repr__()}, 'rb') as pf:
   sheet_vars = pickle.load(pf)
"""
        init_code = init_code + "\n\n" + pickle_code
    if (init_code is not None) and (len(init_code) > 0):
        assert isinstance(init_code, str)
        nb = prepend_code_cell_to_notebook(nb, code_text=f"\n\n{init_code}\n\n")
    html_name = os.path.basename(notebook_file_name)
    html_name = html_name.removesuffix(suffix)
    exec_note = ""
    if output_suffix is not None:
        assert isinstance(output_suffix, str)
        html_name = html_name + output_suffix
        exec_note = f'"{output_suffix}"'
    html_name = html_name + ".html"
    try:
        os.remove(html_name)
    except FileNotFoundError:
        pass
    caught = None
    trace = None
    html_body = ""
    if verbose:
        print(
            f'start render_as_html "{notebook_file_name}" {exec_note} {datetime.datetime.now()}'
        )
    try:
        if kernel_name is not None:
            ep = OurExecutor(timeout=timeout, kernel_name=kernel_name)
        else:
            ep = OurExecutor(timeout=timeout)
        nb_res, nb_resources = ep.preprocess(nb)
        html_exporter = HTMLExporter(exclude_input=exclude_input)
        html_body, html_resources = html_exporter.from_notebook_node(nb_res)
        caught = ep.caught_exception
    except Exception as e:
        caught = e
        trace = traceback.format_exc()
    if exclude_input and (prompt_strip_regexp is not None):
        # strip output prompts
        html_body = re.sub(prompt_strip_regexp, " ", html_body)
    with open(html_name, "wt", encoding="utf-8") as f:
        f.write(html_body)
        if caught is not None:
            f.write("\n<pre>\n")
            f.write(escape_ansi(str(caught)))
            f.write("\n</pre>\n")
        if trace is not None:
            f.write("\n<pre>\n")
            f.write(escape_ansi(str(trace)))
            f.write("\n</pre>\n")
    nw = datetime.datetime.now()
    if tmp_path is not None:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
    if caught is not None:
        if verbose:
            print(
                f'\n\n\texception in render_as_html "{notebook_file_name}" {nw} {escape_ansi(str(caught))}\n\n'
            )
            if trace is not None:
                print(f"\n\n\t\ttrace {escape_ansi(str(trace))}\n\n")
        raise caught
    if verbose:
        print(f'\tdone render_as_html "{notebook_file_name}" {nw}')


_jtask_comparison_attributes = [
    "sheet_name",
    "output_suffix",
    "exclude_input",
    "init_code",
    "path_prefix",
]


class JTask:
    def __init__(
        self,
        sheet_name: str,
        *,
        output_suffix: Optional[str] = None,
        exclude_input: bool = True,
        sheet_vars=None,
        init_code: Optional[str] = None,
        path_prefix: Optional[str] = None,
        strict: bool = True,
    ) -> None:
        """
        Create a Jupyter task.

        :param sheet_name: name of sheet to run can be .ipynb or .py, and suffix can be omitted.
        :param output_suffix: optional string to append to rendered HTML file name.
        :param exclude_input: if True strip input cells out of HTML render.
        :param sheet_vars: if not None value is de-serialized as a variable named "sheet_vars"
        :param init_code: optional code to insert at the top of the Jupyter sheet, used to pass parameters.
        :param path_prefix: optional prefix to add to sheet_name to find Jupyter source.
        :param strict: if True check paths path_prefix and path_prefix/sheetname[.py|.ipynb] exist.
        """
        assert isinstance(sheet_name, str)
        assert isinstance(output_suffix, (str, type(None)))
        assert isinstance(exclude_input, bool)
        assert isinstance(init_code, (str, type(None)))
        assert isinstance(path_prefix, (str, type(None)))
        if strict:
            path = sheet_name
            if (isinstance(path_prefix, str)) and (len(path_prefix) > 9):
                assert os.path.exists(path_prefix)
                path = os.path.join(path_prefix, sheet_name)
            if path.endswith(".py"):
                path = path.removesuffix(".py")
            if path.endswith(".ipynb"):
                path = path.removesuffix(".ipynb")
            py_exists = os.path.exists(path + ".py")
            ipynb_exists = os.path.exists(path + ".ipynb")
            assert (py_exists + ipynb_exists) >= 1
            if (not sheet_name.endswith(".py")) and (not sheet_name.endswith(".ipynb")):
                # no suffix, so must be unambiguous
                assert (py_exists + ipynb_exists) == 1
        self.sheet_name = sheet_name
        self.output_suffix = output_suffix
        self.exclude_input = exclude_input
        self.sheet_vars = sheet_vars
        self.init_code = init_code
        self.path_prefix = path_prefix

    def render_as_html(self) -> None:
        """
        Render Jupyter notebook or Python (treated as notebook) to HTML.
        """
        path = self.sheet_name
        if isinstance(self.path_prefix, str) and (len(self.path_prefix) > 0):
            path = os.path.join(self.path_prefix, self.sheet_name)
        render_as_html(
            path,
            exclude_input=self.exclude_input,
            output_suffix=self.output_suffix,
            sheet_vars=self.sheet_vars,
            init_code=self.init_code,
        )

    def render_py_txt(self) -> None:
        """
        Render Python to text (without nbconvert, nbformat Jupyter bindings)
        """
        path = self.sheet_name
        if isinstance(self.path_prefix, str) and (len(self.path_prefix) > 0):
            path = os.path.join(self.path_prefix, self.sheet_name)
        execute_py(
            path,
            output_suffix=self.output_suffix,
            sheet_vars=self.sheet_vars,
            init_code=self.init_code,
        )

    def __getitem__(self, item):
        return getattr(self, item)

    def _is_valid_operand(self, other):
        return isinstance(other, JTask)

    def __str__(self) -> str:
        args_str = ",\n".join(
            [
                f" {v}={repr(self[v])}"
                for v in _jtask_comparison_attributes + ["sheet_vars"]
            ]
        )
        return "JTask(\n" + args_str + ",\n)"

    def __repr__(self) -> str:
        return self.__str__()


def job_fn(arg: JTask):
    """
    Function to run a JTask job for Jupyter notebook. Exceptions pass through
    """
    assert isinstance(arg, JTask)
    # render notebook
    return arg.render_as_html()


def job_fn_eat_exception(arg: JTask):
    """
    Function to run a JTask job for Jupyter notebook, catching any exception and returning it as a value
    """
    assert isinstance(arg, JTask)
    # render notebook
    try:
        return arg.render_as_html()
    except AssertionError:
        _, _, tb = sys.exc_info()
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]
        print(f"Assertion failed {filename}: {line} (caller {func}) in statement {text}")
    except Exception as e:
        print(f"{arg} caught {escape_ansi(e)}")
        return (arg, e)


def job_fn_py_txt(arg: JTask):
    """
    Function to run a JTask job for Python to txt. Exceptions pass through
    """
    assert isinstance(arg, JTask)
    # render Python
    return arg.render_py_txt()


def job_fn_py_txt_eat_exception(arg: JTask):
    """
    Function to run a JTask job for Python to txt, catching any exception and returning it as a value
    """
    assert isinstance(arg, JTask)
    # render Python
    try:
        return arg.render_py_txt()
    except AssertionError:
        _, _, tb = sys.exc_info()
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]
        print(f"Assertion failed {filename}: {line} (caller {func}) in statement {text}")
    except Exception as e:
        print(f"{arg} caught {escape_ansi(e)}")
        return (arg, e)


def run_pool(
    tasks: Iterable,
    *,
    njobs: int = 4,
    verbose: bool = True,
    stop_on_error: bool = True,
    use_Jupyter: bool = True,
) -> List:
    """
    Run a pool of tasks.

    :param tasks: iterable of tasks
    :param njobs: degree of parallelism
    :param verbose: if True, print on failure
    :param stop_on_error: if True, stop pool on error
    :param use_Jupyter: if True, use nbconvert, nbformat Jupyter fns.
    """
    tasks = list(tasks)
    assert isinstance(njobs, int)
    assert njobs > 0
    assert isinstance(verbose, bool)
    assert isinstance(stop_on_error, bool)
    assert isinstance(use_Jupyter, bool)
    if len(tasks) <= 0:
        return
    for task in tasks:
        assert isinstance(task, JTask)
    res = None
    if stop_on_error:
        # # complex way, allowing a stop on job failure
        # https://stackoverflow.com/a/25791961/6901725
        if use_Jupyter:
            fn = job_fn
        else:
            fn = job_fn_py_txt
        with Pool(njobs) as pool:
            try:
                res = list(
                    pool.imap_unordered(fn, tasks)
                )  # list is forcing iteration over tasks for side-effects
            except Exception:
                if verbose:
                    sys.stdout.flush()
                    print("!!! run_pool: a worker raised an Exception, aborting...")
                    sys.stdout.flush()
                pool.close()
                pool.terminate()
                raise  # re-raise Exception
            else:
                pool.close()
                pool.join()
    else:
        if use_Jupyter:
            fn = job_fn_eat_exception
        else:
            fn = job_fn_py_txt_eat_exception
        # simple way, but doesn't exit until all jobs succeed or fail
        with Pool(njobs) as pool:
            res = list(pool.map(fn, tasks))
    return res


@contextmanager
def declare_task_variables(
    env,
    *,
    result_map: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Copy env["sheet_vars"][k] into env[k] for all k in sheet_vars.keys() if "sheet_vars" defined in env.
    Only variables that are first assigned in the with block of this task manager are allowed to be assigned.
    Setting `env=globals()` works if calling environment assignments are assigning into globals, as (from `help(globals)`): "NOTE: Updates to this dictionary *will* affect 
    name lookups in the current global scope and vice-versa."
    Setting `env=locals()` is not to be trusted as (from `help(locals`)): "NOTE: Whether or not updates to this dictionary will affect name lookups in
    the local scope and vice-versa is *implementation dependent* and not
    covered by any backwards compatibility guarantees."
    Because of the above `declare_task_variables()` is unlikely to work inside a function body.

    :param env: working environment, setting to globals() is usually the correct choice.
    :param result_map: empty dictionary to return results in. result_map["sheet_vars"] is the dictionary if incoming assignments, result_map["declared_vars"] is the dictionary of default names and values.
    :return None:
    """
    sheet_vars = dict()
    pre_known_vars = set()
    if env is not None:
        pre_known_vars = set(env.keys())
    if result_map is not None:
        result_map["sheet_vars"] = sheet_vars
        result_map["declared_vars"] = set()
    if "sheet_vars" in pre_known_vars:
        sheet_vars = env["sheet_vars"]
        if result_map is not None:
            result_map["sheet_vars"] = sheet_vars
        already_assigned_vars = set(sheet_vars.keys()).intersection(pre_known_vars)
        if len(already_assigned_vars) > 0:
            raise ValueError(
                f"declare_task_variables(): attempting to set pre-with variables: {sorted(already_assigned_vars)}"
            )
    try:
        yield
    finally:
        post_known_vars = set(env.keys())
        declared_vars = post_known_vars - pre_known_vars
        if result_map is not None:
            result_map["declared_vars"] = {k: env[k] for k in declared_vars}
        if "sheet_vars" in pre_known_vars:
            unexpected_vars = set(sheet_vars.keys()) - declared_vars
            if len(unexpected_vars) > 0:
                raise ValueError(
                    f"declare_task_variables(): attempting to assign undeclared variables: {sorted(unexpected_vars)}"
                )
            # do the assignments
            for k, v in sheet_vars.items():
                env[k] = v
