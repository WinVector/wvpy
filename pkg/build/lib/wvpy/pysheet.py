
# run with:
#    python -m wvpy.pysheet -in test.py -out test.ipynb
import argparse
import sys
import os
from wvpy.jtools import convert_py_file_to_notebook


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert between .py and .ipynb")
    parser.add_argument('-in', dest='input_file_name', type=str, nargs=1,
        help='name of input file')
    parser.add_argument('-out', dest='output_file_name', type=str, nargs=1,
        help='name of output file')
    args = parser.parse_args()
    input_file_name = args.input_file_name[0]
    output_file_name= args.output_file_name[0]
    assert input_file_name != output_file_name
    assert os.path.exists(input_file_name)
    if input_file_name.endswith('.py'):
        assert output_file_name.endswith('.ipynb')
        convert_py_file_to_notebook(
            py_file=input_file_name,
            ipynb_file=output_file_name,
        )
    elif input_file_name.endswith('.ipynb'):
        assert output_file_name.endswith('.py')
        raise ValueError("not implemented yet")
    else:
        raise ValueError("input file must end with .py or .ipynb")
    return 0


if __name__ == '__main__':
    try:
        ret = main()
    except Exception as ex:
        print(ex)
        sys.exit(-1)
    sys.exit(ret)
