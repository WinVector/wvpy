import data_algebra.test_util
import pandas as pd
import wvpy.util


def test_onehot():
    d = pd.DataFrame({
        'xc': ['a', 'b', 'b'],
        'xd': [1, 1, 2],  # force re-encoding
        'xn': [1.0, 2.0, 3.0],
    })

    enc_bundle = wvpy.util.fit_onehot_enc(d, categorical_var_names=['xc', 'xd'])
    res = wvpy.util.apply_onehot_enc(d, encoder_bundle=enc_bundle)

    expect = pd.DataFrame({
        'xn': [1.0, 2.0, 3.0],
        'xc_a': [1.0, 0.0, 0.0],
        'xc_b': [0.0, 1.0, 1.0],
        'xd_1': [1.0, 1.0, 0.0],
        'xd_2': [0.0, 0.0, 1.0],
        })

    assert data_algebra.test_util.equivalent_frames(res, expect)
