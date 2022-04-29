"""
Utility functions for teaching data science.
"""

from typing import Dict, Iterable, List, Tuple

import re
import os
import numpy
import statistics
import matplotlib
import matplotlib.pyplot
import seaborn
import sklearn
import sklearn.metrics
import sklearn.preprocessing
import itertools
import pandas
import math
from data_algebra.cdata import RecordMap, RecordSpecification


def types_in_frame(d: pandas.DataFrame) -> Dict[str, List[type]]:
    """
    Report what type as seen as values in a Pandas data frame.
    
    :param d: Pandas data frame to inspect, not altered.
    :return: dictionary mapping column names to order lists of types found in column.
    """
    assert isinstance(d, pandas.DataFrame)
    type_dict_map = {
        col_name: {str(type(v)): type(v) for v in d[col_name]}
            for col_name in d.columns
    }
    type_dict = {
        col_name: [type_set[k] for k in sorted(list(type_set.keys()))]
            for col_name, type_set in type_dict_map.items()
    }
    return type_dict


# noinspection PyPep8Naming
def cross_predict_model(
    fitter, X: pandas.DataFrame, y: pandas.Series, plan: List
) -> numpy.ndarray:
    """
    train a model y~X using the cross validation plan and return predictions

    :param fitter: sklearn model we can call .fit() on
    :param X: explanatory variables, pandas DataFrame
    :param y: dependent variable, pandas Series
    :param plan: cross validation plan from mk_cross_plan()
    :return: vector of simulated out of sample predictions
    """

    assert isinstance(X, pandas.DataFrame)
    assert isinstance(y, pandas.Series)
    assert isinstance(plan, List)
    preds = None
    for pi in plan:
        model = fitter.fit(X.iloc[pi["train"], :], y.iloc[pi["train"]])
        predg = model.predict(X.iloc[pi["test"], :])
        # patch results in
        if preds is None:
            preds = numpy.asarray([None] * X.shape[0], dtype=numpy.asarray(predg).dtype)
        preds[pi["test"]] = predg
    return preds


# noinspection PyPep8Naming
def cross_predict_model_proba(
    fitter, X: pandas.DataFrame, y: pandas.Series, plan: List
) -> pandas.DataFrame:
    """
    train a model y~X using the cross validation plan and return probability matrix

    :param fitter: sklearn model we can call .fit() on
    :param X: explanatory variables, pandas DataFrame
    :param y: dependent variable, pandas Series
    :param plan: cross validation plan from mk_cross_plan()
    :return: matrix of simulated out of sample predictions
    """

    assert isinstance(X, pandas.DataFrame)
    assert isinstance(y, pandas.Series)
    assert isinstance(plan, List)
    preds = None
    for pi in plan:
        model = fitter.fit(X.iloc[pi["train"], :], y.iloc[pi["train"]])
        predg = model.predict_proba(X.iloc[pi["test"], :])
        # patch results in
        if preds is None:
            preds = numpy.zeros((X.shape[0], predg.shape[1]))
        for j in range(preds.shape[1]):
            preds[pi["test"], j] = predg[:, j]
    preds = pandas.DataFrame(preds)
    preds.columns = list(fitter.classes_)
    return preds


def mean_deviance(predictions, istrue, *, eps=1.0e-6):
    """
    compute per-row deviance of predictions versus istrue

    :param predictions: vector of probability preditions
    :param istrue: vector of True/False outcomes to be predicted
    :param eps: how close to zero or one we clip predictions
    :return: vector of per-row deviances
    """

    istrue = numpy.asarray(istrue)
    predictions = numpy.asarray(predictions)
    mass_on_correct = numpy.where(istrue, predictions, 1 - predictions)
    mass_on_correct = numpy.maximum(mass_on_correct, eps)
    return -2 * sum(numpy.log(mass_on_correct)) / len(istrue)


def mean_null_deviance(istrue, *, eps=1.0e-6):
    """
    compute per-row nulll deviance of predictions versus istrue

    :param istrue: vector of True/False outcomes to be predicted
    :param eps: how close to zero or one we clip predictions
    :return: mean null deviance of using prevalence as the prediction.
    """

    istrue = numpy.asarray(istrue)
    p = numpy.zeros(len(istrue)) + numpy.mean(istrue)
    return mean_deviance(predictions=p, istrue=istrue, eps=eps)


def mk_cross_plan(n: int, k: int) -> List:
    """
    Randomly split range(n) into k train/test groups such that test groups partition range(n).

    :param n: integer > 1
    :param k: integer > 1
    :return: list of train/test dictionaries

    Example:

    import wvpy.util

    wvpy.util.mk_cross_plan(10, 3)
    """
    grp = [i % k for i in range(n)]
    numpy.random.shuffle(grp)
    plan = [
        {
            "train": [i for i in range(n) if grp[i] != j],
            "test": [i for i in range(n) if grp[i] == j],
        }
        for j in range(k)
    ]
    return plan


# https://win-vector.com/2020/09/13/why-working-with-auc-is-more-powerful-than-one-might-think/
def matching_roc_area_curve(auc: float) -> dict:
    """
    Find an ROC curve with a given area with form of y = 1 - (1 - (1 - x) ** q) ** (1 / q).

    :param auc: area to match
    :return: dictionary of ideal x, y series matching area
    """
    step = 0.01
    eval_pts = numpy.arange(0, 1 + step, step)
    q_eps = 1e-6
    q_low = 0.0
    q_high = 1.0
    while q_low + q_eps < q_high:
        q_mid = (q_low + q_high) / 2.0
        q_mid_area = numpy.mean(1 - (1 - (1 - eval_pts) ** q_mid) ** (1 / q_mid))
        if q_mid_area <= auc:
            q_high = q_mid
        else:
            q_low = q_mid
    q = (q_low + q_high) / 2.0
    return {
        "auc": auc,
        "q": q,
        "x": 1 - eval_pts,
        "y": 1 - (1 - (1 - eval_pts) ** q) ** (1 / q),
    }


# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def plot_roc(
    prediction,
    istrue,
    title="Receiver operating characteristic plot",
    *,
    truth_target=True,
    ideal_line_color=None,
    extra_points=None,
    show=True,
):
    """
    Plot a ROC curve of numeric prediction against boolean istrue.

    :param prediction: column of numeric predictions
    :param istrue: column of items to predict
    :param title: plot title
    :param truth_target: value to consider target or true.
    :param ideal_line_color: if not None, color of ideal line
    :param extra_points: data frame of additional point to annotate graph, columns fpr, tpr, label
    :param show: logical, if True call matplotlib.pyplot.show()
    :return: calculated area under the curve, plot produced by call.

    Example:

    import pandas
    import wvpy.util

    d = pandas.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [False, False, True, True, False]
    })

    wvpy.util.plot_roc(
        prediction=d['x'],
        istrue=d['y'],
        ideal_line_color='lightgrey'
    )

    wvpy.util.plot_roc(
        prediction=d['x'],
        istrue=d['y'],
        ideal_line_color='lightgrey',
        extra_points=pandas.DataFrame({
            'tpr': [0, 1],
            'fpr': [0, 1],
            'label': ['AAA', 'BBB']
        })
    )
    """
    prediction = numpy.asarray(prediction)
    istrue = numpy.asarray(istrue) == truth_target
    fpr, tpr, _ = sklearn.metrics.roc_curve(istrue, prediction)
    auc = sklearn.metrics.auc(fpr, tpr)
    ideal_curve = None
    if ideal_line_color is not None:
        ideal_curve = matching_roc_area_curve(auc)
    matplotlib.pyplot.figure()
    lw = 2
    matplotlib.pyplot.gcf().clear()
    fig1, ax1 = matplotlib.pyplot.subplots()
    ax1.set_aspect("equal")
    matplotlib.pyplot.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = {0:0.2f})" "".format(auc),
    )
    matplotlib.pyplot.fill_between(fpr, tpr, color="orange", alpha=0.3)
    matplotlib.pyplot.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    if extra_points is not None:
        matplotlib.pyplot.scatter(extra_points.fpr, extra_points.tpr, color="red")
        if "label" in extra_points.columns:
            tpr = extra_points.tpr.to_list()
            fpr = extra_points.fpr.to_list()
            label = extra_points.label.to_list()
            for i in range(extra_points.shape[0]):
                txt = label[i]
                if txt is not None:
                    ax1.annotate(txt, (fpr[i], tpr[i]))
    if ideal_curve is not None:
        matplotlib.pyplot.plot(
            ideal_curve["x"], ideal_curve["y"], linestyle="--", color=ideal_line_color
        )
    matplotlib.pyplot.xlim([0.0, 1.0])
    matplotlib.pyplot.ylim([0.0, 1.0])
    matplotlib.pyplot.xlabel("False Positive Rate (1-Specificity)")
    matplotlib.pyplot.ylabel("True Positive Rate (Sensitivity)")
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.legend(loc="lower right")
    if show:
        matplotlib.pyplot.show()
    return auc


def dual_density_plot(
    probs,
    istrue,
    title="Double density plot",
    *,
    truth_target=True,
    positive_label="positive examples",
    negative_label="negative examples",
    ylabel="density of examples",
    xlabel="model score",
    show=True,
):
    """
    Plot a dual density plot of numeric prediction probs against boolean istrue.

    :param probs: vector of numeric predictions.
    :param istrue: truth vector
    :param title: title of plot
    :param truth_target: value considerd true
    :param positive_label=label for positive class
    :param negative_label=label for negative class
    :param ylabel=y axis label
    :param xlabel=x axis label
    :param show: logical, if True call matplotlib.pyplot.show()
    :return: None

    Example:

    import pandas
    import wvpy.util

    d = pandas.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [False, False, True, True, False]
    })

    wvpy.util.dual_density_plot(
        probs=d['x'],
        istrue=d['y'],
    )
    """
    probs = numpy.asarray(probs)
    istrue = numpy.asarray(istrue) == truth_target
    matplotlib.pyplot.gcf().clear()
    preds_on_positive = [
        probs[i] for i in range(len(probs)) if istrue[i] == truth_target
    ]
    preds_on_negative = [
        probs[i] for i in range(len(probs)) if not istrue[i] == truth_target
    ]
    seaborn.kdeplot(preds_on_positive, label=positive_label, shade=True)
    seaborn.kdeplot(preds_on_negative, label=negative_label, shade=True)
    matplotlib.pyplot.ylabel(ylabel)
    matplotlib.pyplot.xlabel(xlabel)
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.legend()
    if show:
        matplotlib.pyplot.show()


def dual_hist_plot(probs, istrue, title="Dual Histogram Plot", *, truth_target=True, show=True):
    """
    plot a dual histogram plot of numeric prediction probs against boolean istrue

    :param probs: vector of numeric predictions.
    :param istrue: truth vector
    :param title: title of plot
    :param truth_target: value to consider in class
    :param show: logical, if True call matplotlib.pyplot.show()
    :return: None

    Example:

    import pandas
    import wvpy.util

    d = pandas.DataFrame({
        'x': [.1, .2, .3, .4, .5],
        'y': [False, False, True, True, False]
    })

    wvpy.util.dual_hist_plot(
        probs=d['x'],
        istrue=d['y'],
    )
    """
    probs = numpy.asarray(probs)
    istrue = numpy.asarray(istrue) == truth_target
    matplotlib.pyplot.gcf().clear()
    pf = pandas.DataFrame({"prob": probs, "istrue": istrue})
    g = seaborn.FacetGrid(pf, row="istrue", height=4, aspect=3)
    bins = numpy.arange(0, 1.1, 0.1)
    g.map(matplotlib.pyplot.hist, "prob", bins=bins)
    matplotlib.pyplot.title(title)
    if show:
        matplotlib.pyplot.show()


def dual_density_plot_proba1(
    probs,
    istrue,
    title="Double density plot",
    *,
    truth_target=True,
    positive_label="positive examples",
    negative_label="negative examples",
    ylabel="density of examples",
    xlabel="model score",
    show=True,
):
    """
    Plot a dual density plot of numeric prediction probs[:,1] against boolean istrue.

    :param probs: matrix of numeric predictions (as returned from predict_proba())
    :param istrue: truth target
    :param title: title of plot
    :param truth_target: value considered true
    :param positive_label=label for positive class
    :param negative_label=label for negative class
    :param ylabel=y axis label
    :param xlabel=x axis label
    :param show: logical, if True call matplotlib.pyplot.show()
    :return: None

    Example:

    d = pandas.DataFrame({
        'x': [.1, .2, .3, .4, .5],
        'y': [False, False, True, True, False]
    })
    d['x0'] = 1 - d['x']
    pmat = numpy.asarray(d.loc[:, ['x0', 'x']])

    wvpy.util.dual_density_plot_proba1(
        probs=pmat,
        istrue=d['y'],
    )
    """
    istrue = numpy.asarray(istrue)
    probs = numpy.asarray(probs)
    matplotlib.pyplot.gcf().clear()
    preds_on_positive = [
        probs[i, 1] for i in range(len(probs)) if istrue[i] == truth_target
    ]
    preds_on_negative = [
        probs[i, 1] for i in range(len(probs)) if not istrue[i] == truth_target
    ]
    seaborn.kdeplot(preds_on_positive, label=positive_label, shade=True)
    seaborn.kdeplot(preds_on_negative, label=negative_label, shade=True)
    matplotlib.pyplot.ylabel(ylabel)
    matplotlib.pyplot.xlabel(xlabel)
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.legend()
    if show:
        matplotlib.pyplot.show()


def dual_hist_plot_proba1(probs, istrue, *, show=True):
    """
    plot a dual histogram plot of numeric prediction probs[:,1] against boolean istrue

    :param probs: vector of probability predictions
    :param istrue: vector of ground truth to condition on
    :param show: logical, if True call matplotlib.pyplot.show()
    :return: None

    Example:

    d = pandas.DataFrame({
        'x': [.1, .2, .3, .4, .5],
        'y': [False, False, True, True, False]
    })
    d['x0'] = 1 - d['x']
    pmat = numpy.asarray(d.loc[:, ['x0', 'x']])

    wvpy.util.dual_hist_plot_proba1(
        probs=pmat,
        istrue=d['y'],
    )
    """
    istrue = numpy.asarray(istrue)
    probs = numpy.asarray(probs)
    matplotlib.pyplot.gcf().clear()
    pf = pandas.DataFrame(
        {"prob": [probs[i, 1] for i in range(probs.shape[0])], "istrue": istrue}
    )
    g = seaborn.FacetGrid(pf, row="istrue", height=4, aspect=3)
    bins = numpy.arange(0, 1.1, 0.1)
    g.map(matplotlib.pyplot.hist, "prob", bins=bins)
    if show:
        matplotlib.pyplot.show()


def gain_curve_plot(prediction, outcome, title="Gain curve plot", *, show=True):
    """
    plot cumulative outcome as a function of prediction order (descending)

    :param prediction: vector of numeric predictions
    :param outcome: vector of actual values
    :param title: plot title
    :param show: logical, if True call matplotlib.pyplot.show()
    :return: None

    Example:

    d = pandas.DataFrame({
        'x': [.1, .2, .3, .4, .5],
        'y': [0, 0, 1, 1, 0]
    })

    wvpy.util.gain_curve_plot(
        prediction=d['x'],
        outcome=d['y'],
    )
    """

    df = pandas.DataFrame(
        {
            "prediction": numpy.array(prediction).copy(),
            "outcome": numpy.array(outcome).copy(),
        }
    )

    # compute the gain curve
    df.sort_values(["prediction"], ascending=[False], inplace=True)
    df["fraction_of_observations_by_prediction"] = (
        numpy.arange(df.shape[0]) + 1.0
    ) / df.shape[0]
    df["cumulative_outcome"] = df["outcome"].cumsum()
    df["cumulative_outcome_fraction"] = df["cumulative_outcome"] / numpy.max(
        df["cumulative_outcome"]
    )

    # compute the wizard curve
    df.sort_values(["outcome"], ascending=[False], inplace=True)
    df["fraction_of_observations_by_wizard"] = (
        numpy.arange(df.shape[0]) + 1.0
    ) / df.shape[0]

    df["cumulative_outcome_by_wizard"] = df["outcome"].cumsum()
    df["cumulative_outcome_fraction_wizard"] = df[
        "cumulative_outcome_by_wizard"
    ] / numpy.max(df["cumulative_outcome_by_wizard"])

    seaborn.lineplot(
        x="fraction_of_observations_by_wizard",
        y="cumulative_outcome_fraction_wizard",
        color="gray",
        linestyle="--",
        data=df,
    )

    seaborn.lineplot(
        x="fraction_of_observations_by_prediction",
        y="cumulative_outcome_fraction",
        data=df,
    )

    seaborn.lineplot(x=[0, 1], y=[0, 1], color="red")
    matplotlib.pyplot.xlabel("fraction of observations by sort criterion")
    matplotlib.pyplot.ylabel("cumulative outcome fraction")
    matplotlib.pyplot.title(title)
    if show:
        matplotlib.pyplot.show()


def lift_curve_plot(prediction, outcome, title="Lift curve plot", *, show=True):
    """
    plot lift as a function of prediction order (descending)

    :param prediction: vector of numeric predictions
    :param outcome: vector of actual values
    :param title: plot title
    :param show: logical, if True call matplotlib.pyplot.show()
    :return: None

    Example:

    d = pandas.DataFrame({
        'x': [.1, .2, .3, .4, .5],
        'y': [0, 0, 1, 1, 0]
    })

    wvpy.util.lift_curve_plot(
        prediction=d['x'],
        outcome=d['y'],
    )
    """

    df = pandas.DataFrame(
        {
            "prediction": numpy.array(prediction).copy(),
            "outcome": numpy.array(outcome).copy(),
        }
    )

    # compute the gain curve
    df.sort_values(["prediction"], ascending=[False], inplace=True)
    df["fraction_of_observations_by_prediction"] = (
        numpy.arange(df.shape[0]) + 1.0
    ) / df.shape[0]
    df["cumulative_outcome"] = df["outcome"].cumsum()
    df["cumulative_outcome_fraction"] = df["cumulative_outcome"] / numpy.max(
        df["cumulative_outcome"]
    )

    # move to lift
    df["lift"] = (
        df["cumulative_outcome_fraction"] / df["fraction_of_observations_by_prediction"]
    )
    seaborn.lineplot(x="fraction_of_observations_by_prediction", y="lift", data=df)
    matplotlib.pyplot.axhline(y=1, color="red")
    matplotlib.pyplot.title(title)
    if show:
        matplotlib.pyplot.show()


# https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
def search_grid(inp: dict) -> List:
    """
    build a cross product of all named dictionary entries

    :param inp: dictionary of value lists
    :return: list of value dictionaries
    """

    gen = (dict(zip(inp.keys(), values)) for values in itertools.product(*inp.values()))
    return [ci for ci in gen]


def grid_to_df(grid: List) -> pandas.DataFrame:
    """
    convert a search_grid list of maps to a pandas data frame

    :param grid: list of combos
    :return: data frame with one row per combo
    """

    n = len(grid)
    keys = [ki for ki in grid[1].keys()]
    return pandas.DataFrame({ki: [grid[i][ki] for i in range(n)] for ki in keys})


def eval_fn_per_row(f, x2, df: pandas.DataFrame) -> List:
    """
    evaluate f(row-as-map, x2) for rows in df

    :param f: function to evaluate
    :param x2: extra argument
    :param df: data frame to take rows from
    :return: list of evaluations
    """

    assert isinstance(df, pandas.DataFrame)
    return [f({k: df.loc[i, k] for k in df.columns}, x2) for i in range(df.shape[0])]


def perm_score_vars(d: pandas.DataFrame, istrue, model, modelvars: List[str], k=5):
    """
    evaluate model~istrue on d permuting each of the modelvars and return variable importances

    :param d: data source (copied)
    :param istrue: y-target
    :param model: model to evaluate
    :param modelvars: names of variables to permute
    :param k: number of permutations
    :return: score data frame
    """

    d2 = d[modelvars].copy()
    d2.reset_index(inplace=True, drop=True)
    istrue = numpy.asarray(istrue)
    preds = model.predict_proba(d2[modelvars])
    basedev = mean_deviance(preds[:, 1], istrue)

    def perm_score_var(victim):
        """Permutation score column named victim"""
        dorig = numpy.array(d2[victim].copy())
        dnew = numpy.array(d2[victim].copy())

        def perm_score_var_once():
            """apply fn once, used for list comprehension"""
            numpy.random.shuffle(dnew)
            d2[victim] = dnew
            predsp = model.predict_proba(d2[modelvars])
            permdev = mean_deviance(predsp[:, 1], istrue)
            return permdev

        # noinspection PyUnusedLocal
        devs = [perm_score_var_once() for rep in range(k)]
        d2[victim] = dorig
        return numpy.mean(devs), statistics.stdev(devs)

    stats = [perm_score_var(victim) for victim in modelvars]
    vf = pandas.DataFrame({"var": modelvars})
    vf["importance"] = [di[0] - basedev for di in stats]
    vf["importance_dev"] = [di[1] for di in stats]
    vf.sort_values(by=["importance"], ascending=False, inplace=True)
    vf = vf.reset_index(inplace=False, drop=True)
    return vf


def threshold_statistics(
    d: pandas.DataFrame, *, model_predictions: str, yvalues: str, y_target=True
) -> pandas.DataFrame:
    """
    Compute a number of threshold statistics of how well model predictions match a truth target.

    :param d: pandas.DataFrame to take values from
    :param model_predictions: name of predictions column
    :param yvalues: name of truth values column
    :param y_target: value considered to be true
    :return: summary statistic frame, include before and after pseudo-observations

    Example:

    import pandas
    import wvpy.util

    d = pandas.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [False, False, True, True, False]
    })

    wvpy.util.threshold_statistics(
        d,
        model_predictions='x',
        yvalues='y',
    )
    """
    # make a thin frame to re-sort for cumulative statistics
    sorted_frame = pandas.DataFrame(
        {"threshold": d[model_predictions].copy(), "truth": d[yvalues] == y_target}
    )
    sorted_frame["orig_index"] = sorted_frame.index + 0
    sorted_frame.sort_values(
        ["threshold", "orig_index"], ascending=[False, True], inplace=True
    )
    sorted_frame.reset_index(inplace=True, drop=True)
    sorted_frame["notY"] = 1 - sorted_frame["truth"]  # falses
    sorted_frame["one"] = 1
    del sorted_frame["orig_index"]

    # pseudo-observation to get end-case (accept nothing case)
    eps = 1.0e-6
    sorted_frame = pandas.concat(
        [
            pandas.DataFrame(
                {
                    "threshold": [sorted_frame["threshold"].max() + eps],
                    "truth": [False],
                    "notY": [0],
                    "one": [0],
                }
            ),
            sorted_frame,
            pandas.DataFrame(
                {
                    "threshold": [sorted_frame["threshold"].min() - eps],
                    "truth": [False],
                    "notY": [0],
                    "one": [0],
                }
            ),
        ]
    )
    sorted_frame.reset_index(inplace=True, drop=True)

    # basic cumulative facts
    sorted_frame["count"] = sorted_frame["one"].cumsum()  # predicted true so far
    sorted_frame["fraction"] = sorted_frame["count"] / max(1, sorted_frame["one"].sum())
    sorted_frame["precision"] = sorted_frame["truth"].cumsum() / sorted_frame[
        "count"
    ].clip(lower=1)
    sorted_frame["true_positive_rate"] = sorted_frame["truth"].cumsum() / max(
        1, sorted_frame["truth"].sum()
    )
    sorted_frame["false_positive_rate"] = sorted_frame["notY"].cumsum() / max(
        1, sorted_frame["notY"].sum()
    )
    sorted_frame["true_negative_rate"] = (
        sorted_frame["notY"].sum() - sorted_frame["notY"].cumsum()
    ) / max(1, sorted_frame["notY"].sum())
    sorted_frame["false_negative_rate"] = (
        sorted_frame["truth"].sum() - sorted_frame["truth"].cumsum()
    ) / max(1, sorted_frame["truth"].sum())
    sorted_frame["accuracy"] = (
        sorted_frame["truth"].cumsum()  # true positive count
        + sorted_frame["notY"].sum()
        - sorted_frame["notY"].cumsum()  # true negative count
    ) / sorted_frame["one"].sum()

    # approximate cdf work
    sorted_frame["cdf"] = 1 - sorted_frame["fraction"]

    # derived facts and synonyms
    sorted_frame["recall"] = sorted_frame["true_positive_rate"]
    sorted_frame["sensitivity"] = sorted_frame["recall"]
    sorted_frame["specificity"] = 1 - sorted_frame["false_positive_rate"]

    # re-order for neatness
    sorted_frame["new_index"] = sorted_frame.index.copy()
    sorted_frame.sort_values(["new_index"], ascending=[False], inplace=True)
    sorted_frame.reset_index(inplace=True, drop=True)

    # clean up
    del sorted_frame["notY"]
    del sorted_frame["one"]
    del sorted_frame["new_index"]
    del sorted_frame["truth"]
    return sorted_frame


def threshold_plot(
    d: pandas.DataFrame,
    pred_var: str,
    truth_var: str,
    truth_target: bool = True,
    threshold_range: Iterable[float] = (-math.inf, math.inf),
    plotvars: Iterable[str] = ("precision", "recall"),
    title: str = "Measures as a function of threshold",
    *,
    show: bool = True,
) -> None:
    """
    Produce multiple facet plot relating the performance of using a threshold greater than or equal to
    different values at predicting a truth target.

    :param d: pandas.DataFrame to plot
    :param pred_var: name of column of numeric predictions
    :param truth_var: name of column with reference truth
    :param truth_target: value considered true
    :param threshold_range: x-axis range to plot
    :param plotvars: list of metrics to plot, must come from ['threshold', 'count', 'fraction',
        'true_positive_rate', 'false_positive_rate', 'true_negative_rate', 'false_negative_rate',
        'precision', 'recall', 'sensitivity', 'specificity', 'accuracy']
    :param title: title for plot
    :param show: logical, if True call matplotlib.pyplot.show()
    :return: None, plot produced as a side effect

    Example:

    import pandas
    import wvpy.util

    d = pandas.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [False, False, True, True, False]
    })

    wvpy.util.threshold_plot(
        d,
        pred_var='x',
        truth_var='y',
        plotvars=("sensitivity", "specificity"),
    )
    """
    if isinstance(plotvars, str):
        plotvars = [plotvars]
    else:
        plotvars = list(plotvars)
    assert isinstance(plotvars, list)
    assert len(plotvars) > 0
    assert all([isinstance(v, str) for v in plotvars])
    threshold_range = list(threshold_range)
    assert len(threshold_range) == 2
    frame = d[[pred_var, truth_var]].copy()
    frame.reset_index(inplace=True, drop=True)
    frame["outcol"] = frame[truth_var] == truth_target

    prt_frame = threshold_statistics(
        frame, model_predictions=pred_var, yvalues="outcol",
    )
    bad_plot_vars = set(plotvars) - set(prt_frame.columns)
    if len(bad_plot_vars) > 0:
        raise ValueError(
            "allowed plotting variables are: "
            + str(prt_frame.columns)
            + ", "
            + str(bad_plot_vars)
            + " unexpected."
        )

    selector = (threshold_range[0] <= prt_frame.threshold) & (
        prt_frame.threshold <= threshold_range[1]
    )
    to_plot = prt_frame.loc[selector, :]

    if len(plotvars) > 1:
        reshaper = RecordMap(
            blocks_out=RecordSpecification(
                pandas.DataFrame({"measure": plotvars, "value": plotvars}),
                control_table_keys=["measure"],
                record_keys=["threshold"],
            )
        )
        prtlong = reshaper.transform(to_plot)
        grid = seaborn.FacetGrid(
            prtlong, row="measure", row_order=plotvars, aspect=2, sharey=False
        )
        grid = grid.map(matplotlib.pyplot.plot, "threshold", "value")
        grid.set(ylabel=None)
        matplotlib.pyplot.subplots_adjust(top=0.9)
        grid.fig.suptitle(title)
    else:
        # can plot off primary frame
        seaborn.lineplot(
            data=to_plot, x="threshold", y=plotvars[0],
        )
        matplotlib.pyplot.suptitle(title)
        matplotlib.pyplot.title(f"measure = {plotvars[0]}")

    if show:
        matplotlib.pyplot.show()


def fit_onehot_enc(
    d: pandas.DataFrame, *, categorical_var_names: Iterable[str]
) -> dict:
    """
    Fit a sklearn OneHot Encoder to categorical_var_names columns.
    Note: we suggest preferring vtreat ( https://github.com/WinVector/pyvtreat ) over this example code.

    :param d: training data
    :param categorical_var_names: list of column names to learn transform from
    :return: encoding bundle dictionary, see apply_onehot_enc() for use.
    """
    assert isinstance(d, pandas.DataFrame)
    assert not isinstance(
        categorical_var_names, str
    )  # single name, should be in a list
    categorical_var_names = list(categorical_var_names)  # clean copy
    assert numpy.all([isinstance(v, str) for v in categorical_var_names])
    assert len(categorical_var_names) > 0
    enc = sklearn.preprocessing.OneHotEncoder(
        categories="auto", drop=None, sparse=False, handle_unknown="ignore"  # default
    )
    enc.fit(d[categorical_var_names])
    produced_column_names = list(enc.get_feature_names_out())
    # return the structure
    encoder_bundle = {
        "categorical_var_names": categorical_var_names,
        "enc": enc,
        "produced_column_names": produced_column_names,
    }
    return encoder_bundle


def apply_onehot_enc(d: pandas.DataFrame, *, encoder_bundle: dict) -> pandas.DataFrame:
    """
    Apply a one hot encoding bundle to a data frame.

    :param d: input data frame
    :param encoder_bundle: transform specification, built by fit_onehot_enc()
    :return: transformed data frame
    """
    assert isinstance(d, pandas.DataFrame)
    assert isinstance(encoder_bundle, dict)
    # one hot re-code columns, preserving column names info
    one_hotted = pandas.DataFrame(
        encoder_bundle["enc"].transform(d[encoder_bundle["categorical_var_names"]])
    )
    one_hotted.columns = encoder_bundle["produced_column_names"]
    # copy over non-invovled columns
    cat_set = set(encoder_bundle["categorical_var_names"])
    complementary_columns = [c for c in d.columns if c not in cat_set]
    res = pandas.concat([d[complementary_columns], one_hotted], axis=1)
    return res


# https://stackoverflow.com/a/56695622/6901725
# from https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])
