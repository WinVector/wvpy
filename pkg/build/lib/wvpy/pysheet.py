
# run with:
#    python -m wvpy.pysheet test.py
#    python -m wvpy.pysheet test.ipynb

import argparse
import os
import sys
import traceback
from wvpy.jtools import convert_py_file_to_notebook, convert_notebook_file_to_py


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert between .py and .ipynb")
    parser.add_argument('infile', metavar='infile', type=str, nargs=1,
        help='name of input file')
    args = parser.parse_args()
    assert len(args.infile) == 1
    input_file_name = args.infile[0]
    assert isinstance(input_file_name, str)
    assert os.path.exists(input_file_name)
    if input_file_name.endswith('.py'):
        output_file_name = input_file_name.removesuffix('.py') + '.ipynb'
        convert_py_file_to_notebook(
            py_file=input_file_name,
            ipynb_file=output_file_name,
        )
    elif input_file_name.endswith('.ipynb'):
        output_file_name = input_file_name.removesuffix('.ipynb') + '.py'
        convert_notebook_file_to_py(
            ipynb_file=input_file_name,
            py_file=output_file_name)
    else:
        raise ValueError("input file name must end with .py or .ipynb")
    return 0


if __name__ == '__main__':
    try:
        ret = main()
        sys.exit(ret)
    except AssertionError:
        _, _, tb = sys.exc_info()
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]
        print(f'Assertion failed {filename}:{line} (caller {func}) in statement {text}')
    except Exception as ex:
        print(ex)
    sys.exit(-1)
    
