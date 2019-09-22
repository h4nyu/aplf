import lightgbm as lgbm
from lightgbm import Dataset
from multiprocessing import cpu_count
from ..eval import r2
import pickle


def eval(preds, train_data):
    loss = r2(preds, train_data.get_label())
    return 'r2', loss, 3


def train(
    path: str,
    train_df,
    tr_indices,
    val_indices,
) -> None:
    tr_df = train_df.iloc[tr_indices]
    val_df = train_df.iloc[val_indices]
    tr_set = Dataset(
        tr_df.drop('Score', axis=1), 
        tr_df['Score']
    )
    val_set = Dataset(
        val_df.drop('Score', axis=1), 
        val_df['Score']
    )
    params = {
        'boosting': 'dart',
        'objective': 'regression',
        'learning_rate': 0.01,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.7,
        'num_leaves': 41,
        'metric': 'mse',
        'drop_rate': 0.15,
        "num_threads":cpu_count(),
    }
    model = lgbm.train(
        train_set=tr_set,
        params=params,
        valid_sets=[tr_set, val_set],
        valid_names=['Train', 'Test'],
        num_boost_round=200000,
        feval=eval,
        early_stopping_rounds=2000,
        verbose_eval=50
    )

    with open(path, 'wb') as f:
        pickle.dump(model, f)
