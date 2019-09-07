from .data import read_csv, TakedaDataset, kfold, create_dataset, save_model, load_model, TakedaPredDataset, save_submit
from torch.utils.data import Subset
from cytoolz.curried import reduce
import typing as t
from .models import Model
from .train.lgbm import train
from logging import getLogger
import pickle
logger = getLogger("takeda.app")


def run(
    prefix:str,
    n_splits: int,
    fold_idx: int,
    lgbm_params,
) -> None:
    df = read_csv('/store/takeda/train.csv')
    indices = kfold(df, n_splits=n_splits)
    tr_set = create_dataset(df.iloc[indices[fold_idx][0]])
    val_set = create_dataset(df.iloc[indices[fold_idx][1]])
    train(
        tr_set,
        val_set,
        lgbm_params,
        path=f"/store/{prefix}-lgbm-model-{n_splits}-{fold_idx}.pkl"
    )


def submit(paths:t.List[str]) -> None:
    df = read_csv('/store/takeda/test.csv')
    models = []
    for p in paths:
        with open(p, 'rb') as f:
            models.append(pickle.load(f))

    preds = [
        model.predict(df)
        for model
        in models
    ]
    preds = reduce(lambda x, y: x+y)(preds)/len(preds)

    save_submit(
        df,
        preds,
        '/store/submit.csv'
    )
