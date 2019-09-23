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
    get_corr_mtrx,
    get_ignore_columns,
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
             loop.run_in_executor(pool,
                 get_corr_mtrx,
                 tr_df.drop('Score', axis=1),
                 f'{base_dir}/tr_corr.pkl'
             ),
             loop.run_in_executor(pool,
                 get_corr_mtrx,
                 ev_df,
                 f'{base_dir}/ev_corr.pkl'
             )
        )
        correlation_threshold = 0.98
        indices = kfold(tr_df, n_splits=n_splits)
        tr_indices, val_indices = indices[fold_idx]
        ignore_columns = get_ignore_columns(tr_corr, correlation_threshold)

        train(
            f"{base_dir}/model-{n_splits}-{fold_idx}.pkl",
            tr_df,
            tr_indices,
            val_indices,
            ignore_columns,
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
