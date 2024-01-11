
"""Python rendering tools"""

import datetime
import os
import pickle
import tempfile
from io import StringIO
from contextlib import redirect_stdout
import traceback

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
    Exceptions raised in the rendering notebook are allowed to pass trough.

    :param source_file_name: name of source file, must end with .ipynb or .py (or type gotten from file system)
    :param output_suffix: optional name to add to result name
    :param verbose logical, if True print while running
    :param sheet_vars: if not None value is de-serialized as a variable named "sheet_vars"
    :param init_code: Python init code for first cell
    :return: None
    """
    assert isinstance(source_file_name, str)
    # get the input
    assert os.path.exists(source_file_name)
    assert source_file_name.endswith(".py")
    python_source = open(source_file_name).read()
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
    try:
        if verbose:
            print(
                f'start execute_py "{source_file_name}" {exec_note} {datetime.datetime.now()}'
            )
        # https://stackoverflow.com/a/3906390
        res_buffer = StringIO()
        with redirect_stdout(res_buffer):
            exec(
                python_source,
                dict(),
                dict(),
            )
        string_res = res_buffer.getvalue()
        with open(result_file_name, "wt") as f:
            f.write(string_res)
            f.write("\n\n")
    except Exception as e:
        caught = e
        trace = traceback.format_exc()
    nw = datetime.datetime.now()
    if tmp_path is not None:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
    if caught is not None:
        with open(result_file_name, "wt") as f:
            f.write(f'\n\nexception in execute_py "{result_file_name}" {nw}\n\n')
            f.write(str(caught))
            f.write("\n\n")
            f.write(str(trace))
            f.write("\n\n")
        if verbose:
            print(f'\texception in execute_py "{result_file_name}" {nw}')
        raise caught
    if verbose:
        print(f'\tdone execute_py "{result_file_name}" {nw}')
