
import numpy
import statistics
import matplotlib
import matplotlib.pyplot
import seaborn
import sklearn
import sklearn.metrics
import itertools
import pandas
import math
from data_algebra.cdata import *


# noinspection PyPep8Naming
def cross_predict_model(fitter, X: pandas.DataFrame, Y: pandas.Series, plan):
    """train a model Y~X using the cross validation plan and return predictions"""
    preds = [None] * X.shape[0]
    for g in range(len(plan)):
        pi = plan[g]
        model = fitter.fit(X.iloc[pi["train"]], Y.iloc[pi["train"]])
        predg = model.predict(X.iloc[pi["test"]])
        for i in range(len(pi["test"])):
            preds[pi["test"][i]] = predg[i]
    return preds


# noinspection PyPep8Naming
def cross_predict_model_prob(fitter, X: pandas.DataFrame, Y: pandas.Series, plan):
    """train a model Y~X using the cross validation plan and return probability matrix"""
    preds = numpy.zeros((X.shape[0], 2))
    for g in range(len(plan)):
        pi = plan[g]
        model = fitter.fit(X.iloc[pi["train"]], Y.iloc[pi["train"]])
        predg = model.predict_proba(X.iloc[pi["test"]])
        for i in range(len(pi["test"])):
            preds[pi["test"][i], 0] = predg[i, 0]
            preds[pi["test"][i], 1] = predg[i, 1]
    return preds


def mean_deviance(predictions, istrue, *, eps=1.0e-6):
    """compute per-row deviance of predictions versus istrue"""
    predictions = [v for v in predictions]
    predictions = numpy.maximum(predictions, eps)
    predictions = numpy.minimum(predictions, 1-eps)
    istrue = [v for v in istrue]
    mass_on_correct = [
        predictions[i] if istrue[i] else 1.0 - predictions[i]
        for i in range(len(istrue))
    ]
    return -2 * sum(numpy.log(mass_on_correct)) / len(istrue)


def mean_null_deviance(istrue, *, eps=1.0e-6):
    """compute per-row nulll deviance of predictions versus istrue"""
    istrue = [v for v in istrue]
    p = numpy.mean(istrue)
    p = numpy.maximum(p, eps)
    p = numpy.minimum(p, 1-eps)
    mass_on_correct = [p if istrue[i] else 1 - p for i in range(len(istrue))]
    return -2 * sum(numpy.log(mass_on_correct)) / len(istrue)


def mk_cross_plan(n: int, k: int):
    """randomly split range(n) into k disjoint groups"""
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


# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def plot_roc(prediction, istrue, title="Receiver operating characteristic plot"):
    """plot a ROC curve of numeric prediction against boolean istrue"""
    prediction = [v for v in prediction]
    istrue = [v for v in istrue]
    fpr, tpr, _ = sklearn.metrics.roc_curve(istrue, prediction)
    auc = sklearn.metrics.auc(fpr, tpr)
    matplotlib.pyplot.figure()
    lw = 2
    matplotlib.pyplot.gcf().clear()
    matplotlib.pyplot.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve  (area = {0:0.2f})" "".format(auc),
    )
    matplotlib.pyplot.fill_between(fpr, tpr, color="orange", alpha=0.3)
    matplotlib.pyplot.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    matplotlib.pyplot.xlim([0.0, 1.0])
    matplotlib.pyplot.ylim([0.0, 1.05])
    matplotlib.pyplot.xlabel("False Positive Rate (1-Specificity)")
    matplotlib.pyplot.ylabel("True Positive Rate (Sensitivity)")
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.legend(loc="lower right")
    matplotlib.pyplot.show()
    return auc


def dual_density_plot(probs, istrue, title="Double density plot"):
    """plot a dual density plot of numeric prediction probs against boolean istrue"""
    probs = [v for v in probs]
    istrue = [v for v in istrue]
    matplotlib.pyplot.gcf().clear()
    preds_on_positive = [probs[i] for i in range(len(probs)) if istrue[i]]
    preds_on_negative = [probs[i] for i in range(len(probs)) if not istrue[i]]
    seaborn.kdeplot(preds_on_positive, label="positive examples", shade=True)
    seaborn.kdeplot(preds_on_negative, label="negative examples", shade=True)
    matplotlib.pyplot.ylabel("density of examples")
    matplotlib.pyplot.xlabel("model score")
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.show()


def dual_density_plot_proba1(probs, istrue):
    """plot a dual density plot of numeric prediction probs[:,1] against boolean istrue"""
    istrue = [v for v in istrue]
    matplotlib.pyplot.gcf().clear()
    preds_on_positive = [probs[i, 1] for i in range(len(probs)) if istrue[i]]
    preds_on_negative = [probs[i, 1] for i in range(len(probs)) if not istrue[i]]
    seaborn.kdeplot(preds_on_positive, label="positive examples", shade=True)
    seaborn.kdeplot(preds_on_negative, label="negative examples", shade=True)
    matplotlib.pyplot.ylabel("density of examples")
    matplotlib.pyplot.xlabel("model score")
    matplotlib.pyplot.show()


def dual_hist_plot_proba1(probs, istrue):
    """plot a dual histogram plot of numeric prediction probs[:,1] against boolean istrue"""
    istrue = [v for v in istrue]
    matplotlib.pyplot.gcf().clear()
    pf = pandas.DataFrame(
        {"prob": [probs[i, 1] for i in range(probs.shape[0])], "istrue": istrue}
    )
    g = seaborn.FacetGrid(pf, row="istrue", height=4, aspect=3)
    bins = numpy.arange(0, 1.1, 0.1)
    g.map(matplotlib.pyplot.hist, "prob", bins=bins)
    matplotlib.pyplot.show()


def gain_curve_plot(prediction, outcome, title="Gain curve plot"):
    """plot cumulative outcome as a function of prediction order (descending)"""
    prediction = [v for v in prediction]
    outcome = [v for v in outcome]
    df = pandas.DataFrame({"prediction": prediction, "outcome": outcome})

    # compute the gain curve
    df.sort_values(["prediction"], ascending=[False], inplace=True)
    df["fraction_of_observations_by_prediction"] = [
        (1 + i) / df.shape[0] for i in range(df.shape[0])
    ]
    df["cumulative_outcome"] = df["outcome"].cumsum()
    df["cumulative_outcome_fraction"] = df["cumulative_outcome"] / numpy.max(
        df["cumulative_outcome"]
    )

    # compute the wizard curve
    df.sort_values(["outcome"], ascending=[False], inplace=True)
    df["fraction_of_observations_by_wizard"] = [
        (1 + i) / df.shape[0] for i in range(df.shape[0])
    ]
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
    matplotlib.pyplot.show()


def lift_curve_plot(prediction, outcome, title="Lift curve plot"):
    """plot lift as a function of prediction order (descending)"""
    prediction = [v for v in prediction]
    outcome = [v for v in outcome]
    df = pandas.DataFrame({"prediction": prediction, "outcome": outcome})
    df.sort_values(["prediction"], ascending=[False], inplace=True)
    df["fraction_of_observations_by_prediction"] = [
        (1 + i) / df.shape[0] for i in range(df.shape[0])
    ]
    df["cumulative_outcome"] = df["outcome"].cumsum()
    df["cumulative_outcome_fraction"] = df["cumulative_outcome"] / numpy.max(
        df["cumulative_outcome"]
    )
    df["lift"] = (
        df["cumulative_outcome_fraction"] / df["fraction_of_observations_by_prediction"]
    )
    seaborn.lineplot(x="fraction_of_observations_by_prediction", y="lift", data=df)
    matplotlib.pyplot.axhline(y=1, color="red")
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.show()


def dual_hist_plot(probs, istrue, title="Dual Histogram Plot"):
    """plot a dual histogram plot of numeric prediction probs against boolean istrue"""
    probs = [v for v in probs]
    istrue = [v for v in istrue]
    matplotlib.pyplot.gcf().clear()
    pf = pandas.DataFrame(
        {"prob": probs, "istrue": istrue}
    )
    g = seaborn.FacetGrid(pf, row="istrue", height=4, aspect=3)
    bins = numpy.arange(0, 1.1, 0.1)
    g.map(matplotlib.pyplot.hist, "prob", bins=bins)
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.show()


# https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
def search_grid(inp):
    """build a cross product of all named dictionary entries"""
    gen = (dict(zip(inp.keys(), values)) for values in itertools.product(*inp.values()))
    return [ci for ci in gen]


def grid_to_df(grid):
    """convert a search_grid list of maps to a pandas data frame"""
    n = len(grid)
    keys = [ki for ki in grid[1].keys()]
    return pandas.DataFrame({ki: [grid[i][ki] for i in range(n)] for ki in keys})


def eval_fn_per_row(f, x2, df):
    """evaluate f(row-as-map, x2) for rows in df"""
    return [f({k: df.loc[i, k] for k in df.columns}, x2) for i in range(df.shape[0])]


def perm_score_vars(d: pandas.DataFrame, istrue, model, modelvars, k=5):
    """evaluate model~istrue on d permuting each of the modelvars and return variable importances"""
    d2 = d[modelvars].copy()
    d2.reset_index(inplace=True, drop=True)
    istrue = [v for v in istrue]
    preds = model.predict_proba(d2[modelvars])
    basedev = mean_deviance(preds[:, 1], istrue)

    def perm_score_var(victim):
        dorig = numpy.array(d2[victim].copy())
        dnew = numpy.array(d2[victim].copy())

        def perm_score_var_once():
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
    return vf


def threshold_statistics(d: pandas.DataFrame, model_predictions, yvalues, *, y_target=True):
    # make a thin frame to re-sort for cumulative statistics
    sorted_frame = pandas.DataFrame({
        'threshold': d[model_predictions].copy(),
        'truth': d[yvalues] == y_target
    })
    sorted_frame['orig_index'] = sorted_frame.index + 0
    sorted_frame.sort_values(['threshold', 'orig_index'], ascending=[False, True], inplace=True)
    sorted_frame.reset_index(inplace=True, drop=True)
    sorted_frame["notY"] = 1 - sorted_frame['truth']  # falses
    sorted_frame["one"] = 1
    del sorted_frame['orig_index']

    # pseudo-observation to get end-case (accept nothing case)
    eps = 1.0e-6
    sorted_frame = pandas.concat([
        pandas.DataFrame({
            'threshold': [sorted_frame['threshold'].max() + eps],
            'truth': [False],
            'notY': [0],
            'one': [0],
        }),
        sorted_frame,
        pandas.DataFrame({
            'threshold': [sorted_frame['threshold'].min() - eps],
            'truth': [False],
            'notY': [0],
            'one': [0],
        }),
    ])
    sorted_frame.reset_index(inplace=True, drop=True)

    # basic cumulative facts
    sorted_frame["count"] = sorted_frame['one'].cumsum()  # predicted true so far
    sorted_frame["fraction"] = sorted_frame["count"] / sorted_frame['one'].sum()
    sorted_frame["precision"] = sorted_frame['truth'].cumsum() / sorted_frame["count"]
    sorted_frame["true_positive_rate"] = sorted_frame['truth'].cumsum() / sorted_frame['truth'].sum()
    sorted_frame["false_positive_rate"] = sorted_frame["notY"].cumsum() / sorted_frame["notY"].sum()
    sorted_frame["true_negative_rate"] = (sorted_frame['notY'].sum() - sorted_frame['notY'].cumsum()) /\
                                         sorted_frame['notY'].sum()
    sorted_frame["false_negative_rate"] = (sorted_frame["truth"].sum() - sorted_frame["truth"].cumsum()) /\
                                          sorted_frame["truth"].sum()
    sorted_frame["enrichment"] = sorted_frame["precision"] / sorted_frame['truth'].mean()

    # derived facts and synonyms
    sorted_frame["gain"] = sorted_frame["enrichment"]
    sorted_frame["lift"] = sorted_frame["gain"] / sorted_frame["fraction"]
    sorted_frame["recall"] = sorted_frame["true_positive_rate"]
    sorted_frame["sensitivity"] = sorted_frame["recall"]
    sorted_frame["specificity"] = 1 - sorted_frame["false_positive_rate"]

    # re-order for neatness
    sorted_frame['new_index'] = sorted_frame.index.copy()
    sorted_frame.sort_values(['new_index'], ascending=[False], inplace=True)
    sorted_frame.reset_index(inplace=True, drop=True)

    # clean up
    del sorted_frame['notY']
    del sorted_frame['one']
    del sorted_frame['new_index']
    del sorted_frame['truth']
    return sorted_frame


def threshold_plot(
    d: pandas.DataFrame,
    pred_var,
    truth_var,
    truth_target=True,
    threshold_range=(-math.inf, math.inf),
    plotvars=("precision", "recall"),
    title="Measures as a function of threshold",
):
    frame = d[[pred_var, truth_var]].copy()
    frame.reset_index(inplace=True, drop=True)
    frame["outcol"] = frame[truth_var] == truth_target

    prt_frame = threshold_statistics(frame, pred_var, "outcol")

    selector = (threshold_range[0] <= prt_frame.threshold) & (
        prt_frame.threshold <= threshold_range[1]
    )
    to_plot = prt_frame.loc[selector, :]

    reshaper = RecordMap(
        blocks_out=RecordSpecification(
            pandas.DataFrame({"measure": plotvars, "value": plotvars}),
            record_keys=["threshold"],
        )
    )

    prtlong = reshaper.transform(to_plot)
    prtlong.head()

    grid = seaborn.FacetGrid(
        prtlong, row="measure", row_order=plotvars, aspect=2, sharey=False
    )
    grid = grid.map(matplotlib.pyplot.plot, "threshold", "value")
    matplotlib.pyplot.subplots_adjust(top=0.9)
    grid.fig.suptitle(title)
    matplotlib.pyplot.show()
