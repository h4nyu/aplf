import typing as t
import lightgbm as lgbm
from lightgbm import Dataset
from multiprocessing import cpu_count
from ..eval import r2
import pickle
from ..data import interpolate


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
    tr_df = interpolate(tr_df)
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
        'learning_rate': 0.05,
        'min_data_in_leaf': 10,
        'feature_fraction': 0.7,
        'num_leaves': 21,
        "max_bin": 128,
        'metric': 'mse',
        'drop_rate': 0.15,
        "num_threads":cpu_count(),
    }
    model = lgbm.train(
        train_set=tr_set,
        params=params,
        valid_sets=[val_set],
        valid_names=['Test'],
        num_boost_round=200000,
        feval=eval,
        early_stopping_rounds=1000,
        verbose_eval=50
    )

    with open(path, 'wb') as f:
        pickle.dump(model, f)

def predict(
    path: str,
    train_df,
) -> t.Any:
    with open(path, mode='rb') as f:
        model = pickle.load(f)
    x_data = train_df.drop('Score', axis=1).values if 'Score' in train_df.columns else train_df.values
    preds = model.predict(x_data)
    return preds

