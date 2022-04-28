
import sys
from wvpy.jtools import convert_py_file_to_notebook


def main() -> int:
    assert len(sys.argv) == 3
    infile = sys.argv[1]
    outfile = sys.argv[2]
    convert_py_file_to_notebook(
        py_file=infile,
        ipynb_file=outfile,
    )
    return 0


if __name__ == '__main__':
    ret = main()
    sys.exit(ret)
