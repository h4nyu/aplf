from aplf.takeda.app import run, submit


def test_run_1() -> None:
    """
    submit 0.615
    local 0.448
    """
    run(base_dir="/store/aug", n_splits=6, fold_idx=1)


def test_submit() -> None:
    submit([
        '/store/-lgbm-model-6-2.pkl',
    ])
