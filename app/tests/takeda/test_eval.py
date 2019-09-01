from aplf.takeda.eval import r2
from torch import arange


def test_r2() -> None:
    x_arr = arange(10, dtype=float)
    y_arr = arange(10, dtype=float)
    assert r2(x_arr, y_arr) == 1.0
