

import re
import datetime
import os
import nbformat
import nbconvert.preprocessors
import pickle
from typing import Optional


def convert_py_code_to_notebook(text: str) -> nbformat.notebooknode.NotebookNode:
    """
    Convert python text to a notebook. 
    "''' begin text" ends any open blocks, and starts a new markdown block
    "''' # end text" ends text, and starts a new code block
    "'''end code'''" ends code blocks, and starts a new code block

    :param text: Python text to convert.
    :return: a notebook 
    """
    # https://stackoverflow.com/a/23729611/6901725
    # https://nbviewer.org/gist/fperez/9716279
    assert isinstance(text, str)
    lines = text.splitlines()
    begin_text_regexp = re.compile(r"^r?'''\s*begin\s+text\s*$")
    end_text_regexp = re.compile(r"^'''\s*#\s*end\s+text\s*$")
    end_code_regexp = re.compile(r"^r?'''\s*end\s+code\s*'''\s*$")
    nbf_v = nbformat.v4
    nb = nbf_v.new_notebook()
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
                txt_block = ('\n'.join(collecting_python)).strip('\n') + '\n'
                if len(txt_block.strip()) > 0:
                    cells.append(nbf_v.new_code_cell(txt_block))
            if (collecting_text is not None) and (len(collecting_text) > 0):
                txt_block = ('\n'.join(collecting_text)).strip('\n') + '\n'
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
    nb['cells'] = cells
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
    nbf_v = nbformat.v4
    nb_out = nbf_v.new_notebook()
    nb_out['cells'] = [nbf_v.new_code_cell(code_text)] + list(nb.cells)
    return nb_out


def convert_py_file_to_notebook(*, py_file: str, ipynb_file: str) -> None:
    """
    Convert python text to a notebook. 
    "''' begin text" ends any open blocks, and starts a new markdown block
    "''' # end text" ends text, and starts a new code block
    "'''end code'''" ends code blocks, and starts a new code block

    :param py_file: Path to python source file.
    :param ipynb_file: Path to notebook result file.
    :return: nothing 
    """
    assert isinstance(py_file, str)
    assert isinstance(ipynb_file, str)
    assert py_file != ipynb_file  # prevent clobber
    with open(py_file, 'r') as f:
        text = f.read()
    nb = convert_py_code_to_notebook(text)
    with open(ipynb_file, 'w') as f:
        nbformat.write(nb, f)


def convert_notebook_code_to_py(nb: nbformat.notebooknode.NotebookNode) -> str:
    """
    Convert ipython notebook inputs to a py code. 
    "''' begin text" ends any open blocks, and starts a new markdown block
    "''' # end text" ends text, and starts a new code block
    "'''end code'''" ends code blocks, and starts a new code block

    :param nb: notebook
    :return: Python source code
    """
    res = []
    code_needs_end = False
    for cell in nb.cells:
        if len(cell.source.strip()) > 0:
            if cell.cell_type == 'code':
                if code_needs_end:
                    res.append("\n'''end code'''\n")
                res.append(cell.source.strip('\n'))
                code_needs_end = True
            else:
                res.append("\n''' begin text")
                res.append(cell.source.strip('\n'))
                res.append("'''  # end text\n")
                code_needs_end = False
    res_text = '\n' + ('\n'.join(res)).strip('\n') + '\n\n'
    return res_text


def convert_notebook_file_to_py(*, ipynb_file: str, py_file: str) -> None:
    """
    Convert ipython notebook inputs to a py file. 
    "''' begin text" ends any open blocks, and starts a new markdown block
    "''' # end text" ends text, and starts a new code block
    "'''end code'''" ends code blocks, and starts a new code block

    :param ipynb_file: Path to notebook input file.
    :param py_file: Path to python result file.
    :return: nothing
    """
    assert isinstance(py_file, str)
    assert isinstance(ipynb_file, str)
    assert py_file != ipynb_file  # prevent clobber
    with open(ipynb_file, "rb") as f:
        nb = nbformat.read(f, as_version=4)
    py_source = convert_notebook_code_to_py(nb)
    with open(py_file, 'w') as f:
        f.write(py_source)        


# https://nbconvert.readthedocs.io/en/latest/execute_api.html
# https://nbconvert.readthedocs.io/en/latest/nbconvert_library.html
# HTML element we are trying to delete: 
#   <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[5]:</div>
def render_as_html(
    notebook_file_name: str,
    *,
    output_suffix: Optional[str] = None,
    timeout:int = 60000,
    kernel_name: Optional[str] = None,
    verbose: bool = True,
    init_code: Optional[str] = None,
    exclude_input: bool = False,
    prompt_strip_regexp: Optional[str] = r'<\s*div\s+class\s*=\s*"jp-OutputPrompt[^<>]*>[^<>]*Out[^<>]*<\s*/div\s*>',
) -> None:
    """
    Render a Jupyter notebook in the current directory as HTML.
    Exceptions raised in the rendering notebook are allowed to pass trough.

    :param notebook_file_name: name of source file, must end with .ipynb or .py
    :param output_suffix: optional name to add to result name
    :param timeout: Maximum time in seconds each notebook cell is allowed to run.
                    passed to nbconvert.preprocessors.ExecutePreprocessor.
    :param kernel_name: Jupyter kernel to use. passed to nbconvert.preprocessors.ExecutePreprocessor.
    :param verbose logical, if True print while running 
    :param init_code: Python init code for first cell
    :param exclude_input: if True, exclude input cells
    :param prompt_strip_regexp: regexp to strip prompts, only used if exclude_input is True
    :return: None
    """
    assert isinstance(notebook_file_name, str)
    if notebook_file_name.endswith(".ipynb"):
        suffix = ".ipynb"
        with open(notebook_file_name, "rb") as f:
            nb = nbformat.read(f, as_version=4)
    elif notebook_file_name.endswith(".py"):
        suffix = ".py"
        with open(notebook_file_name, 'r') as f:
            text = f.read()
        nb = convert_py_code_to_notebook(text)
    else:
        raise ValueError("file name must end with .ipynb or .py")
    if init_code is not None:
        assert isinstance(init_code, str)
        nb = prepend_code_cell_to_notebook(nb, code_text=init_code)
    html_name = notebook_file_name.removesuffix(suffix)
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
    try:
        if verbose:
            print(
                f'start render_as_html "{notebook_file_name}" {exec_note} {datetime.datetime.now()}'
            )
        if kernel_name is not None:
            ep = nbconvert.preprocessors.ExecutePreprocessor(
                timeout=timeout, kernel_name=kernel_name
            )
        else:
            ep = nbconvert.preprocessors.ExecutePreprocessor(timeout=timeout)
        nb_res, nb_resources = ep.preprocess(nb)
        html_exporter = nbconvert.HTMLExporter(exclude_input=exclude_input)
        html_body, html_resources = html_exporter.from_notebook_node(nb_res)
        if exclude_input and (prompt_strip_regexp is not None):
            # strip output prompts
            html_body = re.sub(
                prompt_strip_regexp,
                ' ',
                html_body)
        with open(html_name, "wt") as f:
            f.write(html_body)
    except Exception as e:
        caught = e
    if caught is not None:
        raise caught
    if verbose:
        print(f'\tdone render_as_html "{html_name}" {datetime.datetime.now()}')
