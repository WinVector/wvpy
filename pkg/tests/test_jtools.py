
from wvpy.jtools import (
    pretty_format_python,
    JTask,
)


def test_pretty_format_python():
    res = pretty_format_python("1 +   1")
    assert res.strip() == "1 + 1"


def test_jtask_to_str():
    task = JTask("example_good_notebook.ipynb", strict=False)
    res = str(task)
    assert isinstance(res, str)
