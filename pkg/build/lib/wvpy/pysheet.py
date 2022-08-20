
# run with:
#    python -m wvpy.pysheet test.py
#    python -m wvpy.pysheet test.ipynb

import argparse
import os
import shutil
import sys
import traceback
from wvpy.jtools import convert_py_file_to_notebook, convert_notebook_file_to_py


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert between .py and .ipynb or back (can have suffix, or guess suffix)")
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
    # set up the work request
    base_names_seen = set()
    input_suffices_seen = set()
    tasks = []
    other_suffix = {'.py': '.ipynb', '.ipynb': '.py'}
    for input_file_name in args.infile:
        assert isinstance(input_file_name, str)
        assert len(input_file_name) > 0
        suffix_seen = 'error'  # placeholder/sentinel
        base_name = input_file_name
        if input_file_name.endswith('.py'):
            suffix_seen = '.py'
            base_name = input_file_name.removesuffix(suffix_seen)
        elif input_file_name.endswith('.ipynb'):
            suffix_seen = '.ipynb'
            base_name = input_file_name.removesuffix(suffix_seen)
        else:
            py_exists = os.path.exists(input_file_name + '.py')
            ipynb_exists = os.path.exists(input_file_name + '.ipynb')
            if py_exists == ipynb_exists:
                raise ValueError(f'{base_name}: if no suffix is specified, then exactly one of the .py or ipynb file forms must be present')
            if py_exists:
                suffix_seen = '.py'
            else:
                suffix_seen = '.ipynb'
            input_file_name = input_file_name + suffix_seen
        assert os.path.exists(input_file_name)
        assert suffix_seen in other_suffix.keys()  # expected suffix
        assert base_name not in base_names_seen  # each base file name only used once
        base_names_seen.add(base_name)
        input_suffices_seen.add(suffix_seen)
        if len(input_suffices_seen) != 1:    # only one direction of conversion in batch job
            raise ValueError(f"conversion job may only have one input suffix: {input_suffices_seen}")
        output_file_name = base_name + other_suffix[suffix_seen]
        tasks.append((input_file_name, output_file_name))
    # do the work
    for input_file_name, output_file_name in tasks:
        if not args.quiet:
            print(f'from "{input_file_name}" to "{output_file_name}"')
        # back up result target if present
        if os.path.exists(output_file_name):
            output_backup_file = f'{output_file_name}~'
            if not args.quiet:
                print(f'   copying previous output target "{output_file_name}" to "{output_backup_file}"')
            shutil.copy2(output_file_name, output_backup_file)
        # convert
        if input_file_name.endswith('.py'):
            if not args.quiet:
                print(f"   converting Python {input_file_name} to Jupyter notebook {output_file_name}")
            convert_py_file_to_notebook(
                py_file=input_file_name,
                ipynb_file=output_file_name,
            )
        elif input_file_name.endswith('.ipynb'):
            if not args.quiet:
                print(f'   converting Jupyter notebook "{input_file_name}" to Python "{output_file_name}"')
            convert_notebook_file_to_py(
                ipynb_file=input_file_name,
                py_file=output_file_name)
        else:
            raise ValueError("input file name must end with .py or .ipynb")
        # do any deletions
        if args.delete:
            input_backup_file = f'{input_file_name}~'
            if not args.quiet:
                print(f"   moving input {input_file_name} to {input_backup_file}")
            try:
                os.remove(input_backup_file)
            except FileNotFoundError:
                pass
            os.rename(input_file_name, input_backup_file)
        if not args.quiet:
            print()
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
    
