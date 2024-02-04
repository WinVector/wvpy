"""Classes and functions for working with variable assignments"""

from typing import Any, Dict, Iterable, Optional
from contextlib import contextmanager


def dict_to_assignments_str(values: Dict[str, Any]) -> str:
    """
    Format a dictionary as a block of Python assignment statements.

    Example:
    ```python
    from wvpy.assignment import dict_to_assignments_str
    print(dict_to_assignments_str({"a": 1, "b": 2}))
    ```

    :param values: dictionary
    :return: assignment statements
    """
    assert isinstance(values, dict)
    return (
        "\n"
        + "\n".join([f"{k} = {repr(values[k])}" for k in sorted(values.keys())])
        + "\n"
    )


@contextmanager
def record_assignments(
    env,
    *,
    result: Dict[str, Any],
) -> None:
    """
    Context manager to record all assignments to new variables in a with-block.
    New variables being variables not set prior to entering the with block.
    Setting `env=globals()` works if calling environment assignments are assigning into globals, as (from `help(globals)`): "NOTE: Updates to this dictionary *will* affect 
    name lookups in the current global scope and vice-versa."
    Setting `env=locals()` is not to be trusted as (from `help(locals`)): "NOTE: Whether or not updates to this dictionary will affect name lookups in
    the local scope and vice-versa is *implementation dependent* and not
    covered by any backwards compatibility guarantees."
    Because of the above `record_assignments()` is unlikely to work inside a function body.

    Example:
    ```python
    from wvpy.assignment import dict_to_assignments_str, record_assignments
    assignments = {}
    with record_assignments(globals(), result=assignments):
        a = 1
        b = 2
    print(assignments)
    print(dict_to_assignments_str(assignments))
    ```

    :param env: working environment, setting to `globals()` is usually the correct choice.
    :param result: dictionary to store results in. function calls `.clear()` on result.
    :return None:
    """
    assert isinstance(result, dict)
    pre_known_vars = set(env.keys())
    result.clear()
    try:
        yield
    finally:
        post_known_vars = set(env.keys())
        declared_vars = post_known_vars - pre_known_vars
        for k in sorted(declared_vars):
            result[k] = env[k]


def ensure_names_not_already_assigned(env, *, keys: Iterable[str]) -> None:
    """
    Check that no key in keys is already set in the environment env.
    Raises ValueError if keys are already assigned.
    For an example please see: https://github.com/WinVector/wvpy/blob/main/examples/declare_variables/record_example.ipynb .

    :param env: working environment, setting to `globals()` is usually the correct choice.
    :param keys: keys to confirm not already set.
    :return None:
    """
    already_assigned_vars = set(keys).intersection(set(env.keys()))
    if len(already_assigned_vars) > 0:
        raise ValueError(f"variables already set: {sorted(already_assigned_vars)}")


def assign_values_from_map(
    env, *, values: Dict[str, Any], expected_keys: Optional[Iterable[str]] = None,
) -> None:
    """
    Assign values from map into environment.
    For an example please see: https://github.com/WinVector/wvpy/blob/main/examples/declare_variables/record_example.ipynb .
    Setting `env=globals()` works as (from `help(globals)`): "NOTE: Updates to this dictionary *will* affect 
    name lookups in the current global scope and vice-versa."
    Setting `env=locals()` is not to be trusted as (from `help(locals`)): "NOTE: Whether or not updates to this dictionary will affect name lookups in
    the local scope and vice-versa is *implementation dependent* and not
    covered by any backwards compatibility guarantees."

    :param env: working environment, setting to `globals()` is usually the correct choice.
    :param values: dictionary to copy into environment.
    :param expected_keys: if not null a set of keys we are restricting to
    :return None:
    """
    assert isinstance(values, dict)
    for k in values.keys():
        assert isinstance(k, str)
    if expected_keys is not None:
        unexpected_vars = set(values.keys()) - set(expected_keys)
        if len(unexpected_vars) > 0:
            raise ValueError(
                f"attempting to assign undeclared variables: {sorted(unexpected_vars)}"
            )
    # do the assignments
    for k, v in values.items():
        env[k] = v
