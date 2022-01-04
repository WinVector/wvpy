import pandas

import wvpy.util
import data_algebra.test_util


def test_search_grid():
    res = wvpy.util.search_grid({"a": [1, 2], "b": [3, 4]})
    expect = [{"a": 1, "b": 3}, {"a": 1, "b": 4}, {"a": 2, "b": 3}, {"a": 2, "b": 4}]
    assert res == expect


def test_search_grid_to_df():
    res = wvpy.util.search_grid({"a": [1, 2], "b": [3, 4]})
    res = wvpy.util.grid_to_df(res)
    expect = pandas.DataFrame({"a": [1, 1, 2, 2], "b": [3, 4, 3, 4],})
    assert data_algebra.test_util.equivalent_frames(res, expect)
