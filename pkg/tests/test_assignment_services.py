
import io
from contextlib import redirect_stdout
import ast
import pytest

from wvpy.assignment import (
    assign_values_from_map, 
    dict_to_assignments_str,
    ensure_names_not_already_assigned,
    record_assignments,
)


def test_dict_to_assignments_str():
    result = dict_to_assignments_str({"a": 1, "b": 2})
    assert result.strip() == 'a = 1\nb = 2'


def test_record_assignments_exec():
    # run in an environment where we are assigning to globals()
    with io.StringIO() as strbuf:
        with redirect_stdout(strbuf):
            genv = dict()
            exec(
"""
from wvpy.assignment import record_assignments
assignments = {}
with record_assignments(globals(), result=assignments):
    a = 1
    b = 2
print(assignments)
""",
                genv,
            )
            str_result = strbuf.getvalue().strip()
    assert ast.literal_eval(str_result) == {'a': 1, 'b': 2}


def test_record_assignments_disjoint():
    # execute with fake env to get coverage and confirm empty when assignments are not related to environment
    assignments = dict()
    with record_assignments(dict(), result=assignments):
        a = 1
        b = 2
    assert assignments == dict()


def test_ensure_names_not_already_assigned():
    ensure_names_not_already_assigned(dict(), keys=['x'])
    ensure_names_not_already_assigned(dict(), keys=[])
    ensure_names_not_already_assigned({"x": 1}, keys=[])
    with pytest.raises(ValueError):
        ensure_names_not_already_assigned({"x": 1}, keys=['x'])


def test_assign_values_from_map():
    genv = dict()
    assign_values_from_map(genv, values={"x": 1})
    genv = dict()
    assign_values_from_map(genv, values={"x": 1}, expected_keys=['x', 'z'])
    assert genv == {'x': 1}
    genv = dict()
    with pytest.raises(ValueError):
        assign_values_from_map(genv, values={"x": 1}, expected_keys=['z'])
