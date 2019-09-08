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
from .train.nn import train, pred
from logging import getLogger
import pickle
import pandas as pd
from multiprocessing import Pool
from glob import glob
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


def submit(base_dir: str) -> None:
    ev_df = csv_to_pkl(
        '/store/takeda/test.csv',
        f'{base_dir}/test.pkl',
    )
    ev_dataset = TakedaPredDataset(ev_df)
    model_paths = glob(f'{base_dir}/model-*.pkl')
    models = [
        load_model(p)
        for p
        in model_paths
    ]
    preds = [
        pred(model, ev_dataset)
        for model
        in models
    ]
    preds = reduce(lambda x, y: x+y)(preds)/len(preds)

    save_submit(
        ev_df,
        preds,
        f'{base_dir}/submit.csv'
    )
