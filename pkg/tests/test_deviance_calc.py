import numpy
import wvpy.util


def test_dev_calc_1():
    x = [True, True, False, False]
    p = [0.7, 0.8, 0.2, 0.1]
    dev = wvpy.util.mean_deviance(istrue=x, predictions=p)
    assert dev > 0
    assert abs(0.4541612811124891 - dev) < 1e-3
    null_dev = wvpy.util.mean_null_deviance(x)
    assert dev < null_dev
    assert abs(-2 * numpy.log(0.5) - null_dev) < 1e-3
