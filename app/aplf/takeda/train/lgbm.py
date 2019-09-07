import lightgbm as lgbm
from ..eval import r2
import pickle


def eval(preds, train_data):
    loss = r2(preds, train_data.get_label())
    return 'r2', loss, 3


def train(
    train_set,
    val_set,
    lgbm_params,
    path: str,
) -> None:

    model = lgbm.train(
        train_set=train_set,
        params=lgbm_params,
        valid_sets=[train_set, val_set],
        valid_names=['Train', 'Test'],
        num_boost_round=200000,
        feval=eval,
        early_stopping_rounds=2000,
        verbose_eval=50
    )

    with open(path, 'wb') as f:
        pickle.dump(model, f)
