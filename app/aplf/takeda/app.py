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
    dump_hist_plot,
)
from cytoolz.curried import reduce
from sklearn.metrics import r2_score
import typing as t
from .models import Model
from .train.nn import train, pred
from logging import getLogger
import pandas as pd
from multiprocessing import Pool
from glob import glob
from concurrent.futures import ProcessPoolExecutor
import asyncio
logger = getLogger("takeda.app")


async def explore(
    base_dir:str,
) -> None:
    tr_df = csv_to_pkl(
        '/store/takeda/train.csv',
        f'{base_dir}/train.pkl',
    )
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor(max_workers=12) as pool:
         futures = [
             loop.run_in_executor(
                 pool,
                 dump_hist_plot,
                 tr_df[c],
                 f"{base_dir}/hist-{c}.png"
             )
             for c
             in tr_df.columns
         ]
         asyncio.gather(*futures)

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


def pre_submit(base_dir: str) -> None:
    tr_df = csv_to_pkl(
        '/store/takeda/train.csv',
        f'{base_dir}/train.pkl',
    )
    tr_dataset = TakedaDataset(tr_df)
    model_paths = glob(f'{base_dir}/model-*.pkl')
    logger.info(f"{model_paths}")
    models = [
        load_model(p)
        for p
        in model_paths
    ]
    preds = [
        pred(model, tr_dataset)
        for model
        in models
    ]
    preds = reduce(lambda x, y: x+y)(preds)/len(preds)

    print(r2_score(tr_df['Score'], preds))
    save_submit(
        tr_df,
        preds,
        f'{base_dir}/pre_submit.csv'
    )

def submit(base_dir: str) -> None:
    ev_df = csv_to_pkl(
        '/store/takeda/test.csv',
        f'{base_dir}/test.pkl',
    )
    ev_dataset = TakedaPredDataset(ev_df)
    model_paths = glob(f'{base_dir}/model-*.pkl')
    logger.info(f"{model_paths}")
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
