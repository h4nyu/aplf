from aplf.takeda.eval import r2
import numpy as np


def test_r2() -> None:
    x_arr = np.arange(10)
    y_arr = np.arange(10)
    assert r2(x_arr, y_arr) == 1.0
