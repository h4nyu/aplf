from aplf.takeda.eval import r2
import numpy as np
import pytest
from torch import tensor
from sklearn.metrics import r2_score



@pytest.mark.parametrize("x, y", [
    (
        tensor([1, 4, 2, 1, 2, 3], dtype=float),
        tensor([3, 1, 7, 2, 3, 3],dtype=float),
    ),
])
def test_r2(x, y) -> None:
    res = r2(y, x)
    assert r2_score(x, y) == res
