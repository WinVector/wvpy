
import pandas as pd
from wvpy.util import types_in_frame


def test_types_in_frame():
    d = pd.DataFrame({
        'x': [1, 2],
        'y': ['a', 'b'],
        'z': ['a', 1],
    })
    found = types_in_frame(d)
    expect = {
        'x': [int],
        'y': [str],
        'z': [int, str],
    }
    assert found == expect

