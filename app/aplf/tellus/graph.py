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
from .data import TellusDataset, load_train_df, kfold, load_test_df
from . import train as tra
from .predict import predict
from .preprocess import take_topk, cleanup, cut_bin, add_mask_size, groupby, avarage_dfs, dump_json,  get_segment_indices
import torch
import os



class Graph(object):
    def __init__(self,
                 id,
                 dataset_dir,
                 output_dir,
                 n_splits,
                 base_train_config,
                 folds,
                 train_method,
                 ):
        params = locals()
        torch.manual_seed(0)

        ids = pipe(
            range(n_splits),
            filter(lambda x: x in folds),
            list
        )

        train_df_path = delayed(load_train_df)(
            dataset_dir=os.path.join(dataset_dir, 'train'),
            output=os.path.join(output_dir, 'train.pqt')
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

        model_dirs = pipe(
            zip(ids, train_sets),
            map(lambda x: delayed(getattr(tra, train_method))(
                **base_train_config,
                model_dir=os.path.join(output_dir, f"{id}-fold-{x[0]}"),
                sets=x[1],
                log_dir=f'{config["TENSORBORAD_LOG_DIR"]}/{id}/{x[0]}/base',
            )),
            list
        )

        test_df_path = load_test_df(
            dataset_dir='/store/tellus/test',
            output=os.path.join(output_dir, 'test.pqt')
        )
        test_df = delayed(pd.read_parquet)(test_df_path)
        test_dataset = delayed(TellusDataset)(
            test_df,
            has_y=False,
        )


        submission_df_path = delayed(predict)(
            model_dirs=model_dirs,
            log_dir=f'{config["TENSORBORAD_LOG_DIR"]}/{id}/sub',
            dataset=test_dataset,
            log_interval=10,
            out_path=f'{output_dir}/{id}_submission.tsv',
        )

        self.output = delayed(lambda x: x)((
            model_paths,
            submission_df_path,
        ))

    def __call__(self, *args, **kwargs):
        return self.output.compute(*args, **kwargs)
