from .data import(
    csv_to_pkl, 
    extracet_summary,
    TakedaDataset, 
    kfold, 
    create_dataset, 
    save_model, 
    load_model, 
    TakedaPredDataset, 
    save_submit,
    extract_col_type,
    compare_feature,
)
from cytoolz.curried import reduce
import typing as t
from .models import Model
from .train.nn import train
from logging import getLogger
import pickle
import pandas as pd
from multiprocessing import Pool
logger = getLogger("takeda.app")


def run(
    base_dir:str,
    n_splits: int,
    fold_idx: int,
) -> None:
    tr_df = csv_to_pkl(
        '/store/takeda/train.csv',
        f'{base_dir}/train.pkl',
    )
    ev_df = csv_to_pkl(
        '/store/takeda/test.csv',
        f'{base_dir}/test.pkl',
    )

    feature_df = extracet_summary(
        tr_df,
        f'{base_dir}/tr_feature.json',
    )

    feature_df = extracet_summary(
        ev_df,
        f'{base_dir}/ev_feature.json',
    )

    indices = kfold(tr_df, n_splits=n_splits)
    tr_dataset = TakedaDataset(tr_df)
    ev_dataset = TakedaPredDataset(ev_df)
    tr_indices, val_indices = indices[fold_idx]

    train(
        f"{base_dir}/model-{n_splits}-{fold_idx}.pkl",
        tr_dataset,
        ev_dataset,
        tr_indices,
        val_indices
    )


def submit(paths: t.List[str]) -> None:
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
