
"""Python rendering tools"""

import datetime
import os
from io import StringIO
from contextlib import redirect_stderr, redirect_stdout
import traceback
import copy
from typing import Iterable, List, Optional


def execute_py(
    source_file_name: str,
    *,
    output_suffix: Optional[str] = None,
    verbose: bool = True,
    sheet_vars = None,
    init_code: Optional[str] = None,
) -> None:
    """
    Render a Python file to text.
    Exceptions raised in the rendering notebook are allowed to pass through.

    :param source_file_name: name of source file, must end with .py (.py added if not present)
    :param output_suffix: optional name to add to result name
    :param verbose logical, if True print while running
    :param sheet_vars: if not None value is de-serialized as a variable named "sheet_vars" (usually a dictionary)
    :param init_code: Python init code for first cell
    :return: None
    """
    assert isinstance(source_file_name, str)
    # get the input
    assert not source_file_name.endswith(".ipynb")
    if not source_file_name.endswith(".py"):
        source_file_name = source_file_name + ".py"
    assert source_file_name.endswith(".py")
    assert os.path.exists(source_file_name)
    with open(source_file_name, encoding="utf-8") as inf:
        python_source = inf.read()
    if (init_code is not None) and (len(init_code) > 0):
        assert isinstance(init_code, str)
        python_source = (
            init_code
            + "\n\n"
            + python_source
        )
    result_file_name = os.path.basename(source_file_name)
    result_file_name = result_file_name.removesuffix(".py")
    exec_note = ""
    if output_suffix is not None:
        assert isinstance(output_suffix, str)
        result_file_name = result_file_name + output_suffix
        exec_note = f'"{output_suffix}"'
    result_file_name = result_file_name + ".txt"
    try:
        os.remove(result_file_name)
    except FileNotFoundError:
        pass
    caught = None
    trace = None
    if verbose:
        print(
            f'start execute_py "{source_file_name}" {exec_note} {datetime.datetime.now()}'
        )
    # https://stackoverflow.com/a/3906390
    res_buffer_stdout = StringIO()
    res_buffer_stderr = StringIO()
    exec_env = dict()
    if sheet_vars is not None:
        exec_env["sheet_vars"] = copy.deepcopy(sheet_vars)
    with redirect_stdout(res_buffer_stdout):
        with redirect_stderr(res_buffer_stderr):
            try:
                # https://docs.python.org/3/library/functions.html#exec
                exec(
                    python_source,
                    exec_env,
                )
            except Exception as e:
                caught = e
                trace = traceback.format_exc()
    string_res = res_buffer_stdout.getvalue() + "\n\n" + res_buffer_stderr.getvalue()
    if caught is not None:
        string_res = string_res + "\n\n" + str(caught)
    if trace is not None:
        string_res = string_res + "\n\n" + str(trace)
    with open(result_file_name, "wt", encoding="utf-8") as f:
        f.write(string_res)
        f.write("\n\n")
    nw = datetime.datetime.now()
    if caught is not None:
        if verbose:
            print(f'\texception in execute_py "{source_file_name}" {nw}')
        raise caught
    if verbose:
        print(f'\tdone execute_py "{source_file_name}" {nw}')
