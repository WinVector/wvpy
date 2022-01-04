import pandas
import numpy.random
import sklearn.linear_model
import wvpy.util
import data_algebra.test_util


def test_perm_score_vars():
    numpy.random.seed(2022)
    d = pandas.DataFrame({"y": numpy.random.normal(size=100),})
    for i in range(5):
        vname = f"x_{i}"
        d[vname] = numpy.random.normal(size=d.shape[0])
        d["y"] = d["y"] + d[vname]
    for i in range(5):
        vname = f"n_{i}"
        d[vname] = numpy.random.normal(size=d.shape[0])
    d["y"] = d["y"] > 0.1
    vars = [c for c in d.columns if c != "y"]
    model = sklearn.linear_model.LogisticRegression()
    model.fit(d.loc[:, vars], d["y"])
    scores = wvpy.util.perm_score_vars(
        d=d, model=model, istrue=d["y"], modelvars=vars, k=100,
    )
    scores["signal_variable"] = [v.startswith("x_") for v in scores["var"]]
    worst_good = numpy.min(scores.loc[scores["signal_variable"], "importance"])
    best_bad = numpy.max(
        scores.loc[numpy.logical_not(scores["signal_variable"]), "importance"]
    )
    assert worst_good > best_bad
