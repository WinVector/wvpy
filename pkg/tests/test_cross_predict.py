import numpy
import pandas
import sklearn.linear_model
import wvpy.util


def test_cross_predict_1():
    numpy.random.seed(2022)
    d = pandas.DataFrame({"x": range(10),})
    d["y"] = 2 * d["x"] + 1
    plan = wvpy.util.mk_cross_plan(n=d.shape[0], k=3)
    fitter = sklearn.linear_model.LinearRegression()
    preds_cross = wvpy.util.cross_predict_model(
        fitter=fitter, X=d.loc[:, ["x"]], y=d["y"], plan=plan
    )
    assert numpy.max(numpy.abs(preds_cross - d["y"])) < 1e-5
    fitter.fit(X=d.loc[:, ["x"]], y=d["y"])
    preds_regular = fitter.predict(d.loc[:, ["x"]])
    assert numpy.max(numpy.abs(preds_regular - d["y"])) < 1e-5


def test_cross_predict_proba_1():
    numpy.random.seed(2022)
    d = pandas.DataFrame({"x": numpy.random.normal(size=100),})
    d["y"] = numpy.where(
        (d["x"] + numpy.random.normal(size=d.shape[0])) > 0.0, "b", "a"
    )
    plan = wvpy.util.mk_cross_plan(n=d.shape[0], k=3)
    fitter = sklearn.linear_model.LogisticRegression()
    preds_cross = wvpy.util.cross_predict_model_proba(
        fitter=fitter, X=d.loc[:, ["x"]], y=d["y"], plan=plan
    )
    fitter.fit(X=d.loc[:, ["x"]], y=d["y"])
    preds_regular = fitter.predict_proba(d.loc[:, ["x"]])
    assert numpy.abs(preds_regular - preds_cross).max(axis=0).max() < 0.1
    
