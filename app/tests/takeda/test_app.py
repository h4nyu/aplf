from aplf.takeda.app import run, submit


def test_run_0() -> None:
    """
    submit 0.615
    local 0.448
    """
    lgbm_params = {
        'num_threads': 11,
        'learning_rate': 0.05,
        'num_leaves': 4,
        "max_bin": 64,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'min_data_in_leaf': 40,
        'min_data_in_bin': 5,
    }
    run(prefix='', n_splits=6, fold_idx=2, lgbm_params=lgbm_params)

def test_run_1() -> None:
    """
    submit 0.615
    local 0.448
    """
    lgbm_params = {
        'num_threads': 11,
        'learning_rate': 0.05,
        'num_leaves': 4,
        "max_bin": 64,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'min_data_in_leaf': 40,
        'min_data_in_bin': 5,
    }
    run(prefix='', n_splits=6, fold_idx=1, lgbm_params=lgbm_params)


def test_submit() -> None:
    submit([
        '/store/-lgbm-model-6-2.pkl',
    ])
