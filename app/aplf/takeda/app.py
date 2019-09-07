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
from torch.utils.data import Subset
from cytoolz.curried import reduce
import typing as t
from .models import Model
from .train.lgbm import train
from logging import getLogger
import pickle
import pandas as pd
from multiprocessing import Pool
logger = getLogger("takeda.app")


def run(
    base_dir:str,
    n_splits: int,
    fold_idx: int,
    lgbm_params: t.Dict[str, t.Any],
) -> None:
    train_path = csv_to_pkl(
        '/store/takeda/train.csv',
        f'{base_dir}/train.pkl',
    )
    ev_path = csv_to_pkl(
        '/store/takeda/test.csv',
        f'{base_dir}/test.pkl',
    )
    tr_feature = extracet_summary(
        train_path,
        f'{base_dir}/tr_feature.json',
    )

    ev_feature = extracet_summary(
        ev_path,
        f'{base_dir}/ev_feature.json',
    )

    feature_compare = compare_feature(
        tr_feature,
        ev_feature,
        f'{base_dir}/compare.json',
    )

    feature_compare = compare_feature(
        tr_feature,
        ev_feature,
        f'{base_dir}/compare.json',
    )




    #  indices = kfold(df, n_splits=n_splits)
    #  tr_set = create_dataset(df.iloc[indices[fold_idx][0]])
    #  val_set = create_dataset(df.iloc[indices[fold_idx][1]])
    #  train(
    #      tr_set,
    #      val_set,
    #      lgbm_params,
    #      path=f"/store/{prefix}-lgbm-model-{n_splits}-{fold_idx}.pkl"
    #  )


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
