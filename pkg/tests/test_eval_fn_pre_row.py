import pandas
import wvpy.util


def test_eval_fn_per_row():
    d = pandas.DataFrame({"a": [1, 2], "b": [3, 4],})

    def f(mp, x):
        return mp["a"] + mp["b"] + x

    res = wvpy.util.eval_fn_per_row(f, 7, d)
    expect = [11, 13]
    assert res == expect
