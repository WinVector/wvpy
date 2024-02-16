# run with:
#    python -m wvpy.render_workbook test.py
#    python -m wvpy.render_workbook test.ipynb

from typing import Iterable
import argparse
import os
import sys
import traceback
from wvpy.jtools import render_as_html
from wvpy.ptools import execute_py
from wvpy.util import escape_ansi


def render_workbook(
    infiles: Iterable[str],
    *,
    quiet: bool = False,
    strip_input: bool = True,
    use_Jupyter: bool = True,
    init_code: str = "",
) -> int:
    """
    Render a list of Jupyter notebooks.

    :param infiles: list of file names to process
    :param quiet: if true do the work quietly
    :param strip_input: if true strip input cells and cell numbering
    :param use_Jupyter: if True, use nbconvert, nbformat Jupyter fns.
    :param init_code: workbook initialization code.
    :return: 0 if successful
    """
    # checks
    assert isinstance(quiet, bool)
    assert isinstance(strip_input, bool)
    assert isinstance(use_Jupyter, bool)
    assert isinstance(init_code, str)
    assert not isinstance(infiles, str)  # common error
    infiles = list(infiles)
    assert len(infiles) > 0
    assert len(set(infiles)) == len(infiles)
    tasks = []
    for input_file_name in infiles:
        assert isinstance(input_file_name, str)
        assert len(input_file_name) > 0
        assert not input_file_name.endswith(".html")
        assert not input_file_name.endswith(".pdf")
        if not (input_file_name.endswith(".py") or input_file_name.endswith(".ipynb")):
            py_exists = os.path.exists(input_file_name + ".py")
            ipynb_exists = os.path.exists(input_file_name + ".ipynb")
            if py_exists == ipynb_exists:
                raise ValueError(
                    "if no suffix is specified, then exactly one of the .py or ipynb file forms must be present"
                )
            if py_exists:
                input_file_name = input_file_name + ".py"
            else:
                input_file_name = input_file_name + ".ipynb"
        assert input_file_name.endswith(".py") or input_file_name.endswith(".ipynb")
        assert os.path.exists(input_file_name)
        tasks.append(input_file_name)
    # do the work
    for input_file_name in tasks:
        if use_Jupyter:
            render_as_html(
                input_file_name,
                exclude_input=strip_input,
                verbose=quiet == False,
                init_code=init_code,
            )
        else:
            execute_py(
                input_file_name,
                verbose=quiet == False,
                init_code=init_code,
            )
    return 0


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            prog="wvpy.render_workbook",
            description="Render .py or .ipynb to .html by executing using nbcovert Jupyter",
        )
        parser.add_argument(
            "infile",
            metavar="infile",
            type=str,
            nargs="+",
            help="name of input file(s)",
        )
        parser.add_argument(
            "--strip_input",
            action="store_true",
            help="strip input cells and cell markers",
        )
        parser.add_argument("--quiet", action="store_true", help="quiet operation")
        parser.add_argument(
            "--pytxt",
            action="store_true",
            help="render Python to txt (without nbconvert/Jupyter)",
        )
        parser.add_argument("--init", default="", type=str, help="Python initial code")
        args = parser.parse_args()
        # checks
        assert isinstance(args.quiet, bool)
        assert isinstance(args.strip_input, bool)
        assert isinstance(args.pytxt, bool)
        assert isinstance(args.init, str)
        assert len(args.infile) > 0
        assert len(set(args.infile)) == len(args.infile)
        ret = render_workbook(
            quiet=args.quiet,
            strip_input=args.strip_input,
            infiles=args.infile,
            use_Jupyter=args.pytxt is False,
            init_code=args.init,
        )
        sys.exit(ret)
    except AssertionError:
        _, _, tb = sys.exc_info()
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]
        print(f"Assertion failed {filename}: {line} (caller {func}) in statement {text}")
    except Exception as ex:
        print(escape_ansi(ex))
    sys.exit(-1)
