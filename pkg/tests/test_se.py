
import os
import pytest

from wvpy.util import suppress_stdout_stderr


def test_suppress_stdout_stderr():
    with suppress_stdout_stderr():
        x = 1 + 1  # not much of a test
    assert x == 2

