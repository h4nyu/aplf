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
    get_corr_mtrx,
    save_heatmap,
    get_ignore_columns,
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

async def run(
    base_dir:str,
    n_splits: int,
    fold_idx: int,
) -> None:
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor(max_workers=12) as pool:
        tr_df, ev_df = await asyncio.gather(
             loop.run_in_executor(
                 pool,
                 csv_to_pkl,
                 '/store/takeda/train.csv',
                 f'{base_dir}/train.pkl',
             ),
             loop.run_in_executor(
                 pool,
                 csv_to_pkl,
                 '/store/takeda/test.csv',
                 f'{base_dir}/test.pkl',
             )
        )
        tr_corr, ev_corr = await asyncio.gather(
             loop.run_in_executor(
                 pool,
                 get_corr_mtrx,
                 tr_df.drop('Score', axis=1),
                 f'{base_dir}/tr_corr.pkl'
             ),
             loop.run_in_executor(
                 pool,
                 get_corr_mtrx,
                 ev_df,
                 f'{base_dir}/ev_corr.pkl'
             )
        )
        correlation_threshold = 0.95
        indices = kfold(tr_df, n_splits=n_splits)
        tr_indices, val_indices = indices[fold_idx]
        ignore_columns = get_ignore_columns(tr_corr, correlation_threshold)

        save_heatmap(
            tr_corr[tr_corr > correlation_threshold],
            f'{base_dir}/tr_corr.png'
        )
        tr_dataset = TakedaDataset(
            tr_df.iloc[tr_indices],
            ignore_columns=ignore_columns,
        )
        val_dataset = TakedaDataset(
            tr_df.iloc[val_indices],
            ignore_columns=ignore_columns,
        )
        train(
            f"{base_dir}/model-{n_splits}-{fold_idx}.pkl",
            tr_dataset,
            val_dataset,
        )





def pre_submit(base_dir: str) -> None:
    tr_df = csv_to_pkl(
        '/store/takeda/train.csv',
        f'{base_dir}/train.pkl',
    )
    tr_dataset = TakedaDataset(tr_df)
    model_paths = glob(f'{base_dir}/model-*.pkl')
    logger.info(f"{model_paths}")

    tr_corr = get_corr_mtrx(
        tr_df.drop('Score', axis=1),
        f'{base_dir}/tr_corr.pkl'
    )
    correlation_threshold = 0.95
    ignore_columns = get_ignore_columns(tr_corr, correlation_threshold)
    dataset = TakedaDataset(
        tr_df,
        ignore_columns=ignore_columns,
    )

    models = [
        load_model(p)
        for p
        in model_paths
    ]
    preds = [
        pred(model, dataset)
        for model
        in models
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
