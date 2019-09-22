from aplf.takeda.apps.gbdt import (
    run,
)
import pytest

BASE_DIR="/store/gbdt"
def test_run() -> None:
    """
    submit 0.615
    local 0.448
    """
    run(base_dir=BASE_DIR, n_splits=6, fold_idx=1)


