
# run with:
#    python -m wvpy.render_workbook test.py
#    python -m wvpy.render_workbook test.ipynb

import argparse
import os
import sys
import traceback
from wvpy.jtools import render_as_html


def main() -> int:
    parser = argparse.ArgumentParser(description="Render .py or .ipynb to .html by executing in Jupyter")
    parser.add_argument(
        'infile', 
        metavar='infile', 
        type=str, 
        nargs='+',
        help='name of input file(s)')
    parser.add_argument('--strip_input', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    # checks
    assert isinstance(args.quiet, bool)
    assert isinstance(args.strip_input, bool)
    assert len(args.infile) > 0
    assert len(set(args.infile)) == len(args.infile)
    for input_file_name in args.infile:
        assert isinstance(input_file_name, str)
        assert len(input_file_name) > 0
        assert input_file_name.endswith('.py') or input_file_name.endswith('.ipynb')
        assert os.path.exists(input_file_name)
    # do the work
    for input_file_name in args.infile:
        render_as_html(input_file_name, exclude_input=args.strip_input, verbose=args.quiet == False)
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
    
