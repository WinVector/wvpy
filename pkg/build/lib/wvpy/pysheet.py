
# run with:
#    python -m wvpy.pysheet test.py
#    python -m wvpy.pysheet test.ipynb

import argparse
import os
import sys
import traceback
from wvpy.jtools import convert_py_file_to_notebook, convert_notebook_file_to_py


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert between .py and .ipynb (can have suffix, or guess suffix)")
    parser.add_argument('--quiet', action='store_true', help='delete input file')
    parser.add_argument('--delete', action='store_true', help='delete input file')
    parser.add_argument(
        'infile', 
        metavar='infile', 
        type=str, 
        nargs='+',
        help='name of input file(s)')
    args = parser.parse_args()
    # some pre-checks
    assert len(args.infile) > 0
    assert len(set(args.infile)) == len(args.infile)
    assert isinstance(args.quiet, bool)
    assert isinstance(args.delete, bool)
    input_suffices_seen = set()
    tasks = []
    other_suffix = {'.py': '.ipynb', '.ipynb': '.py'}
    for input_file_name in args.infile:
        assert isinstance(input_file_name, str)
        assert len(input_file_name) > 0
        suffix_seen = 'error'  # placeholder
        if input_file_name.endswith('.py'):
            suffix_seen = '.py'
        elif input_file_name.endswith('.ipynb'):
            suffix_seen = '.ipynb'
        else:
            py_exists = os.path.exists(input_file_name + '.py')
            ipynb_exists = os.path.exists(input_file_name + '.ipynb')
            if py_exists == ipynb_exists:
                raise ValueError("if no suffix is specified, then exactly one of the .py or ipynb file forms must be present")
            if py_exists:
                suffix_seen = '.py'
            else:
                suffix_seen = '.ipynb'
            input_file_name = input_file_name + suffix_seen
        assert suffix_seen in other_suffix.keys()
        input_suffices_seen.add(suffix_seen)
        if len(input_suffices_seen) != 1:
            raise ValueError(f"saw more than one input suffix: {input_suffices_seen}")
        assert os.path.exists(input_file_name)
        output_file_name = input_file_name.removesuffix(suffix_seen) + other_suffix[suffix_seen]
        if os.path.exists(output_file_name):
            if os.path.getmtime(output_file_name) > os.path.getmtime(input_file_name):
                raise ValueError(f"output {output_file_name} is already newer than input f{input_file_name}")
        tasks.append((input_file_name, output_file_name))
    if len(input_suffices_seen) != 1:
        raise ValueError(f"expected only one input suffix: {input_suffices_seen}")
    # do the work
    for input_file_name, output_file_name in tasks:
        if input_file_name.endswith('.py'):
            if not args.quiet:
                print(f"converting {input_file_name} to {output_file_name}")
            convert_py_file_to_notebook(
                py_file=input_file_name,
                ipynb_file=output_file_name,
            )
        elif input_file_name.endswith('.ipynb'):
            if not args.quiet:
                print(f"converting {input_file_name} to {output_file_name}")
            convert_notebook_file_to_py(
                ipynb_file=input_file_name,
                py_file=output_file_name)
        else:
            raise ValueError("input file name must end with .py or .ipynb")
    # do any deletions
    if args.delete:
        for input_file_name, output_file_name in tasks:
            if not args.quiet:
                print(f"removing {input_file_name}")
            os.remove(input_file_name)  # Note: we remove input as we are replacing it with output
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
    
