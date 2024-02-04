

import io
from contextlib import redirect_stdout
import ast

from wvpy.jtools import declare_task_variables


def test_declare_task_variables_exec():
    # run in an environment where we are assigning to globals()
    with io.StringIO() as strbuf:
        with redirect_stdout(strbuf):
            genv = dict()
            exec(
"""
from wvpy.jtools import declare_task_variables
result_map = {}
with declare_task_variables(globals(), result_map=result_map):
    a = 1
    b = 2
print(result_map)
""",
                genv,
            )
            str_result = strbuf.getvalue().strip()
    assert ast.literal_eval(str_result) == {'sheet_vars': {}, 'declared_vars': {'a': 1, 'b': 2}}


def test_declare_task_variables_disjoint():
    # execute with fake env to get coverage and confirm empty when assignments are not related to environment
    result_map = {}
    with declare_task_variables(dict(), result_map=result_map):
        a = 1
        b = 2
    assert result_map['sheet_vars'] == dict()
    assert result_map['declared_vars'] == dict()
