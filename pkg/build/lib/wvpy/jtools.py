

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
            if collecting_python is not None:
                txt_block = '\n'.join(collecting_python) + '\n'
                cells.append(nbf_v.new_code_cell(txt_block))
            if collecting_text is not None:
                txt_block = '\n'.join(collecting_text) + '\n'
                cells.append(nbf_v.new_markdown_cell(txt_block))
            collecting_python = None
            collecting_text = None
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


def convert_py_file_to_notebook(py_file: str, ipynb_file: str) -> None:
    """
    Convert python text to a notebook. 
    "''' begin text" ends any open blocks, and starts a new markdown block
    "''' # end text" ends text, and starts a new code block
    "'''end code'''" ends code blocks, and starts a new code block

    :param py_file: Path to python source file.
    :param ipynb_file: Path to notebook result file.
    :return: a notebook 
    """
    assert isinstance(py_file, str)
    assert isinstance(ipynb_file, str)
    assert py_file != ipynb_file  # prevent clobber
    with open(py_file, 'r') as f:
        text = f.read()
    nb = convert_py_code_to_notebook(text)
    with open(ipynb_file, 'w') as f:
        nbformat.write(nb, f)


# https://stackoverflow.com/a/56695622/6901725
# from https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


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
    parameters = None,
    parameter_file_name: Optional[str] = None,
    exclude_input: bool = False,
    prompt_strip_regexp: Optional[str] = r'<\s*div\s+class\s*=\s*"jp-OutputPrompt[^<>]*>[^<>]*Out[^<>]*<\s*/div\s*>',
) -> None:
    """
    Render a Jupyter notebook in the current directory as HTML.
    Exceptions raised in the rendering notebook are allowed to pass trough.

    :param notebook_file_name: name of source file.
    :param output_suffix: optional name to add to result name
    :param timeout: Maximum time in seconds each notebook cell is allowed to run.
                    passed to nbconvert.preprocessors.ExecutePreprocessor.
    :param kernel_name: Jupyter kernel to use. passed to nbconvert.preprocessors.ExecutePreprocessor.
    :param verbose logical, if True print while running 
    :param parameters: arbitrary object to write to paramater_file_name
    :param parameter_file_name: file name to pickle parameters to
    :param exclude_input: if True, exclude input cells
    :param prompt_strip_regexp: regexp to strip prompts
    :return: None
    """
    assert isinstance(notebook_file_name, str)
    suffix = ".ipynb"
    assert notebook_file_name.endswith(suffix)
    html_name = notebook_file_name[: -len(suffix)]
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
    notebook_file_path = notebook_file_name
    with open(notebook_file_path, "rb") as f:
        nb = nbformat.read(f, as_version=4)
    if parameter_file_name is not None:
        assert isinstance(parameter_file_name, str)
        with open(parameter_file_name, "wb") as f:
            pickle.dump(parameters, f)
    caught = None
    try:
        if verbose:
            print(
                f'start render_as_html "{notebook_file_path}" {exec_note} {datetime.datetime.now()}'
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
        if prompt_strip_regexp is not None:
            # strip output prompts
            html_body = re.sub(
                prompt_strip_regexp,
                ' ',
                html_body)
        with open(html_name, "wt") as f:
            f.write(html_body)
    except Exception as e:
        caught = e
    if parameter_file_name is not None:
        try:
            os.remove(parameter_file_name)
        except FileNotFoundError:
            pass
    if caught is not None:
        raise caught
    if verbose:
        print(f'\tdone render_as_html "{html_name}"" {datetime.datetime.now()}')
