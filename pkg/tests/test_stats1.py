import pandas
import wvpy.util
import data_algebra.test_util
import data_algebra.util


def test_stats1():
    d = pandas.DataFrame({"x": [1, 2, 3, 4, 5], "y": [False, False, True, True, False]})

    stats = wvpy.util.threshold_statistics(d, model_predictions="x", yvalues="y",)
    # print(data_algebra.util.pandas_to_example_str(stats))

    expect = pandas.DataFrame({
        'threshold': [0.999999, 1.0, 2.0, 3.0, 4.0, 5.0, 5.000001],
        'count': [5, 5, 4, 3, 2, 1, 0],
        'fraction': [1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
        'precision': [0.4, 0.4, 0.5, 0.6666666666666666, 0.5, 0.0, 0.0],
        'true_positive_rate': [1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0],
        'false_positive_rate': [1.0, 1.0, 0.6666666666666666, 0.3333333333333333, 0.3333333333333333,
                                0.3333333333333333, 0.0],
        'true_negative_rate': [0.0, 0.0, 0.3333333333333333, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666,
                               1.0],
        'false_negative_rate': [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0],
        'accuracy': [0.4, 0.4, 0.6, 0.8, 0.6, 0.4, 0.6],
        'cdf': [0.0, 0.0, 0.19999999999999996, 0.4, 0.6, 0.8, 1.0],
        'recall': [1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0],
        'sensitivity': [1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0],
        'specificity': [0.0, 0.0, 0.33333333333333337, 0.6666666666666667, 0.6666666666666667, 0.6666666666666667, 1.0],
    })

    assert data_algebra.test_util.equivalent_frames(stats, expect)
