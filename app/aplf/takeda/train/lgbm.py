import lightgbm as lgbm
from ..eval import r2
import pickle

def eval(preds, train_data):
    loss = r2(preds, train_data.get_label())
    return loss, 2, 3

def train(
    train_set,
    val_set,
    lgbm_params,
    path:str,
) -> None:
    params = {
        'boosting': 'dart',
        'objective': 'regression',
        'learning_rate': 0.01,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.7,
        'num_leaves': 41,
        'metric': 'mse',
        'drop_rate': 0.15,
        **lgbm_params,
    }

    model = lgbm.train(
        train_set=train_set,
        params=params,
        valid_sets=[train_set, val_set],
        valid_names=['Train', 'Test'],
        num_boost_round=100000,
        feval=eval,
        early_stopping_rounds=100,
        verbose_eval=20
    )

    with open(path, 'wb') as f:
        pickle.dump(model, f)

