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
from .data import TellusDataset, load_train_df, kfold
from .train import base_train
from .preprocess import take_topk, cleanup, cut_bin, add_mask_size, groupby, avarage_dfs, dump_json,  get_segment_indices

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
        torch.manual_seed(0)

        ids = pipe(
            range(n_splits),
            filter(lambda x: x in folds),
            list
        )

        train_df_path = delayed(load_train_df)(
            dataset_dir=join(dataset_dir, 'train'),
            output=join(output_dir, 'train.pqt')
        )

        train_df = delayed(pd.read_parquet)(train_df_path)

        kfolded = delayed(kfold)(
            train_df,
            n_splits
        )

        train_sets = pipe(
            ids,
            map(lambda x: delayed(lambda i: i[x])(kfolded)),
            list
        )

        model_paths = pipe(
            zip(ids, train_sets),
            map(lambda x: delayed(base_train)(
                **base_train_config,
                model_path=join(output_dir, f"{id}-fold-{x[0]}-base-model.pt"),
                sets=x[1],
                log_dir=f'{config["TENSORBORAD_LOG_DIR"]}/{id}/{x[0]}/base',
            )),
            list
        )

        self.output = delayed(lambda x: x)((
            model_paths,
        ))

    def __call__(self, *args, **kwargs):
        return self.output.compute(*args, **kwargs)
