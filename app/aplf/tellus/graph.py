from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk
from sklearn.model_selection import train_test_split
from dask import delayed
from datetime import datetime
from torch.utils.data import Subset
import numpy as np
import pandas as pd
import uuid
import os
from aplf import config
from .dataset import TellusDataset, load_dataset_df
from .train import base_train
from .preprocess import take_topk, cleanup, cut_bin, add_mask_size, groupby, avarage_dfs, dump_json, kfold, get_segment_indices

from os.path import join

class Graph(object):
    def __init__(self,
                 id,
                 dataset_dir,
                 output_dir,
                 n_splits,
                 base_train_config,
                 top_num,
                 folds,
                 ):
        params = locals()

        ids = pipe(
            range(n_splits),
            list
        )

        train_df_path = delayed(load_dataset_df)(
            dataset_dir=join(dataset_dir, 'train'),
            output=join(output_dir, 'train.pqt')
        )

        train_df = delayed(pd.read_parquet)(train_df_path)
        kfolded = delayed(kfold)(train_df, n_splits)

        dataset = delayed(TellusDataset)(
            train_df,
            has_y=True,
        )

        fold_train_sets = pipe(
            range(n_splits),
            map(lambda idx: delayed(lambda x: x[idx][0])(kfolded)),
            map(lambda x: delayed(Subset)(dataset, x)),
            list
        )

        fold_val_sets = pipe(
            range(n_splits),
            map(lambda idx: delayed(lambda x: x[idx][1])(kfolded)),
            map(lambda x: delayed(Subset)(dataset, x)),
            list
        )

        train_sets = pipe(
            zip(ids, fold_train_sets, fold_val_sets),
            filter(lambda x: x[0] in folds),
            list
        )

        model_paths = pipe(
            train_sets,
            map(lambda x: delayed(base_train)(
                **base_train_config,
                model_path=join(output_dir, f"{id}-fold-{x[0]}-base-model.pt"),
                train_set=x[1],
                val_set=x[2],
                log_dir=f'{config["TENSORBORAD_LOG_DIR"]}/{id}/{x[0]}/base',
            )),
            list
        )


        self.output = delayed(lambda x: x)((
            model_paths,
        ))

    def __call__(self):
        return self.output.compute()


