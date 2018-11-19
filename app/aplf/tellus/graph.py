import dask
from pathlib import Path
from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk
from sklearn.model_selection import train_test_split
from dask import delayed
from datetime import datetime
from torch.utils.data import Subset
import numpy as np
import pandas as pd
import uuid
import os
from aplf.config import TENSORBORAD_LOG_DIR
from .data import TellusDataset, load_train_df, kfold, load_test_df
from .train import train_multi
from .predict import predict
from .preprocess import take_topk, cleanup, cut_bin, add_mask_size, groupby, avarage_dfs, dump_json,  get_segment_indices
import torch
import os
from aplf.flow import Flow


class Graph(Flow):
    def __init__(self, config):
        torch.manual_seed(config['seed'])
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.dataset_dir = Path(config['dataset_dir'])
        self.n_splits = config['n_splits']
        self.fold_indices = pipe(
            range(self.n_splits),
            filter(lambda x: x in config['folds']),
            list
        )

    def dump_config(self):
        dump_json(self.output_dir / Path('config.json'), self.config)

    def create_train_sets(self):
        train_df_path = load_train_df(
            dataset_dir=self.dataset_dir/Path('train'),
            output=self.output_dir / Path('train.pqt')
        )
        train_df = pd.read_parquet(train_df_path)
        kfolded = kfold(
            train_df,
            self.config['n_splits']
        )

        train_sets = pipe(
            self.fold_indices,
            map(lambda x: kfolded[x]),
            list
        )
        return train_sets

    def create_test_set(self):
        test_df_path = load_test_df(
            dataset_dir=self.dataset_dir/Path('test'),
            output=self.output_dir / Path('train.pqt')
        )
        test_df = pd.read_parquet(test_df_path)
        test_dataset = TellusDataset(
            test_df,
            has_y=False,
        )
        return test_dataset

    def flow(self):
        self.output_dir.mkdir(exist_ok=True)
        self.dump_config()
        train_sets = self.create_train_sets()

        model_paths = pipe(
            zip(self.fold_indices, train_sets),
            map(lambda x: delayed(train_multi)(
                **self.config['train_config'],
                model_path=self.output_dir/Path(f"fold-{x[0]}.pt"),
                sets=x[1],
                log_dir=f'{TENSORBORAD_LOG_DIR}/{self.output_dir}/fold-{x[0]}',
            )),
            lambda x: dask.compute(*x)
        )

        test_set = self.create_test_set()
        submission_df_path = predict(
            model_paths=model_paths,
            dataset=test_set,
            out_path=self.output_dir / Path("submission.tsv"),
        )
        return submission_df_path
