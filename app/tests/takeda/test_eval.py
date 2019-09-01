from aplf.takeda.eval import r2
import pytest
from torch import arange


@pytest.mark.parametrize("x, y", [
    (
        arange(1, 11, dtype=float),
        arange(10, dtype=float)
    ),
    (
        arange(20, 30, dtype=float),
        arange(10, dtype=float)
    ),
    (
        -arange(20, 30, dtype=float),
        arange(10, dtype=float)
    ),
])
def test_r2(x, y) -> None:
    res = r2(x, y)
    print(res)
    assert res < 1.0
