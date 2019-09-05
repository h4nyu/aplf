from aplf.takeda.app import run, submit

def test_run_0() -> None:
    """
    submit 0.615
    """
    lgbm_params = {
        'min_data_in_leaf': 20,
        'feature_fraction': 0.7,
        'max_bin_by_feature': 20,
        'learning_rate': 0.01,
        'num_leaves': 10,
        'metric': 'mse',
        'drop_rate': 0.15,
        "max_bin": 50,
        "l2_leaf_reg": 0.01,
    }
    run(n_splits=10, fold_idx=0, lgbm_params=lgbm_params)

def test_run_1() -> None:
    """
    feature_fraction 0.7 -> 0.8
    """
    lgbm_params = {
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'max_bin_by_feature': 20,
        'learning_rate': 0.01,
        'num_leaves': 10,
        'metric': 'mse',
        'drop_rate': 0.15,
        "max_bin": 50,
        "l2_leaf_reg": 0.01,
    }
    run(
        prefix="feature_fraction",
        n_splits=10,
        fold_idx=0,
        lgbm_params=lgbm_params,
    )

def test_submit() -> None:
    submit('/store/lgbm-model-10-0.pkl')
