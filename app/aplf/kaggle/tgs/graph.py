from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk
from sklearn.model_selection import train_test_split
from dask import delayed
from datetime import datetime
from torch.utils.data import Subset
import numpy as np
import pandas as pd
import uuid
from aplf import config
from .dataset import TgsSaltDataset, load_dataset_df
from .train import base_train
from .fine_tune import fine_train
from .predict import predict
from .preprocess import take_topk, cleanup, cut_bin, add_mask_size, groupby, avarage_dfs, dump_json, kfold, get_segment_indices


class Graph(object): 
    def __init__(self,
                 id,
                 dataset_dir,
                 output_dir,
                 n_splits,
                 base_train_config,
                 fine_train_config,
                 top_num,
                 folds,
                 ):
        params = locals()

        ids = pipe(
            range(n_splits),
            list
        )

        dataset_df = delayed(load_dataset_df)(
            dataset_dir,
            'train.csv'
        )
        dataset = delayed(TgsSaltDataset)(
            dataset_df,
            has_y=True,
        )

        kfolded = delayed(kfold)(dataset, n_splits)

        train_sets = pipe(
            range(n_splits),
            map(lambda idx: delayed(lambda x: x[idx][0])(kfolded)),
            map(lambda x: delayed(Subset)(dataset, x)),
            list
        )

        seg_sets = pipe(
            train_sets,
            map(delayed(lambda x: x.indices)),
            map(lambda x: delayed(get_segment_indices)(dataset, x)),
            map(lambda x: delayed(Subset)(dataset, x)),
            list
        )

        val_sets = pipe(
            range(n_splits),
            map(lambda idx: delayed(lambda x: x[idx][1])(kfolded)),
            map(lambda x: delayed(Subset)(dataset, x)),
            list
        )

        predict_dataset_df = delayed(load_dataset_df)(
            dataset_dir,
            'sample_submission.csv'
        )

        predict_set = delayed(TgsSaltDataset)(
            predict_dataset_df,
            has_y=False
        )
        trains = pipe(
            zip(ids, train_sets, seg_sets, val_sets),
            filter(lambda x: x[0] in folds),
            list
        )


        model_paths = pipe(
            trains,
            map(lambda x: delayed(base_train)(
                **base_train_config,
                model_path=f"{output_dir}/id-{id}-fold-{x[0]}-base-model.pt",
                train_set=x[1],
                seg_set=x[2],
                val_set=x[3],
                no_lable_set=predict_set,
                log_dir=f'{config["TENSORBORAD_LOG_DIR"]}/{id}/{x[0]}/base',
            )),
            list
        )

        #  model_paths = pipe(
        #      zip(trains, model_paths),
        #      map(lambda x: delayed(fine_train)(
        #          **fine_train_config,
        #          base_model_path=x[1],
        #          model_path=f"{output_dir}/id-{id}-fold-{x[0][0]}-fine-model.pt",
        #          train_set=x[0][1],
        #          seg_set=x[0][2],
        #          val_set=x[0][3],
        #          no_lable_set=predict_set,
        #          log_dir=f'{config["TENSORBORAD_LOG_DIR"]}/{id}/{x[0][0]}/fine',
        #      )),
        #      list
        #  )

        submission_df = delayed(predict)(
            model_paths=model_paths,
            log_dir=f'{config["TENSORBORAD_LOG_DIR"]}/{id}/sub',
            dataset=predict_set,
            log_interval=10,
        )
        #
        submission_df = delayed(lambda df: df[['rle_mask']])(submission_df)
        submission_file = delayed(lambda df: df.to_csv(f"{output_dir}/id-{id}-submission.csv"))(
            submission_df,
        )

        self.output = delayed(lambda x: x)((
            model_paths,
            submission_df,
            submission_file,
        ))


