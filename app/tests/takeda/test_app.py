from aplf.takeda.app import run, submit, pre_submit

def test_run_0() -> None:
    """
    submit 0.615
    local 0.448
    """
    run(base_dir="/store/aug", n_splits=6, fold_idx=0)

def test_run_1() -> None:
    """
    submit 0.615
    local 0.448
    """
    run(base_dir="/store/aug", n_splits=6, fold_idx=1)


def test_submit() -> None:
    submit(base_dir="/store/aug")

def test_pre_submit() -> None:
    pre_submit(base_dir="/store/aug")
