from ..data import(
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
    dump_hist_plot,
)
from cytoolz.curried import reduce
from sklearn.metrics import r2_score
import typing as t
from ..train.gbdt import train, predict
from logging import getLogger
import pandas as pd
from multiprocessing import Pool
from glob import glob

from concurrent.futures import ProcessPoolExecutor
import asyncio
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

    indices = kfold(tr_df, n_splits=n_splits)
    tr_indices, val_indices = indices[fold_idx]

    train(
        f"{base_dir}/model-{n_splits}-{fold_idx}.pkl",
        tr_df,
        tr_indices,
        val_indices
    )

def pre_submit(base_dir: str) -> None:
    tr_df = csv_to_pkl(
        '/store/takeda/train.csv',
        f'{base_dir}/train.pkl',
    )
    model_paths = glob(f'{base_dir}/model-*.pkl')
    logger.info(f"{model_paths}")
    preds = [
        predict(p, tr_df)
        for p
        in model_paths
    ]
    preds = reduce(lambda x, y: x+y)(preds)/len(preds)

    score = r2_score(tr_df['Score'], preds)
    logger.info(f"{score}")
    save_submit(
        tr_df,
        preds,
        f'{base_dir}/pre_submit.csv'
    )

def submit(base_dir: str) -> None:
    tr_df = csv_to_pkl(
        '/store/takeda/test.csv',
        f'{base_dir}/test.pkl',
    )
    model_paths = glob(f'{base_dir}/model-*.pkl')
    logger.info(f"{model_paths}")
    preds = [
        predict(p, tr_df)
        for p
        in model_paths
    ]
    preds = reduce(lambda x, y: x+y)(preds)/len(preds)
    save_submit(
        tr_df,
        preds,
        f'{base_dir}/submit.csv'
    )
