import wvpy.util


def test_match_auc_1():
    for auc in [0, 0.1, 0.5, 0.9, 1]:
        fit = wvpy.util.matching_roc_area_curve(auc)
        assert abs(fit["auc"] - auc) < 1.0e-3
