from aplf.takeda.app import run, submit, pre_submit, explore
import pytest

BASE_DIR="/store/bottleneck"
def test_run_0() -> None:
    """
    submit 0.615
    local 0.448
    """
    run(base_dir=BASE_DIR, n_splits=6, fold_idx=0)

def test_submit() -> None:
    submit(base_dir=BASE_DIR)

def test_pre_submit() -> None:
    pre_submit(base_dir=BASE_DIR)

@pytest.mark.asyncio
async def test_explore() -> None:
    await explore(base_dir="/store/aug")
