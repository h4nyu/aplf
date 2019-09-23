from aplf.takeda.apps.gbdt import (
    run,
    pre_submit,
    submit,
)
import pytest

BASE_DIR="/store/gbdt"
@pytest.mark.asyncio
async def test_run() -> None:
    """
    submit 0.615
    local 0.448
    """
    await run(base_dir=BASE_DIR, n_splits=6, fold_idx=1)


def test_pre_submit() -> None:
    pre_submit(base_dir=BASE_DIR)

def test_submit() -> None:
    submit(base_dir=BASE_DIR)
