from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk
from sklearn.model_selection import train_test_split
from dask import delayed
from datetime import datetime
import numpy as np
import pandas as pd
import uuid
from .dataset import TgsSaltDataset, load_dataset_df
from .train import train
from .predict import predict
from .preprocess import take_topk, cleanup, cut_bin, add_mask_size, groupby, avarage_dfs


class Graph(object):
    def __init__(self,
                 id,
                 dataset_dir,
                 output_dir,
                 epochs,
                 batch_size,
                 val_split_size,
                 patience,
                 base_size,
                 parallel,
                 top_num,
                 ):

        ids = pipe(
            range(parallel),
            map(lambda x: uuid.uuid4()),
            list
        )

        dataset_df = delayed(load_dataset_df)(dataset_dir, 'train.csv')
        dataset_df = delayed(cleanup)(dataset_df)

        spliteds = pipe(
            ids,
            map(lambda x: delayed(train_test_split)(
                dataset_df,
                test_size=val_split_size,
                shuffle=True
            )),
            list
        )
        train_datasets = pipe(
            spliteds,
            map(delayed(lambda x: x[0])),
            map(lambda x: delayed(TgsSaltDataset)(
                x,
            )),
            list
        )

        val_datasets = pipe(
            spliteds,
            map(delayed(lambda x: x[1])),
            map(delayed(TgsSaltDataset)),
            list
        )

        model_paths = pipe(
            zip(ids, train_datasets, val_datasets),
            map(lambda x: delayed(train)(
                model_id=x[0],
                model_path=f"{output_dir}/model_{x[0]}.pt",
                train_dataset=x[1],
                val_dataset=x[2],
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                base_size=base_size,
                log_dir=f"{id}/{x[0]}",
            )),
            list
        )

        scores = pipe(
            zip(ids, model_paths, val_datasets),
            map(lambda x: delayed(predict)(
                model_paths=[x[1]],
                log_dir=f"{id}/{x[0]}/val",
                dataset=x[2],
                log_interval=1,
            )),
            map(delayed(lambda df: df['score'].mean())),
            list
        )

        top_model_paths = delayed(take_topk)(scores, model_paths, top_num)

        predict_dataset_df = delayed(load_dataset_df)(
            dataset_dir,
            'sample_submission.csv'
        )

        predict_dataset = delayed(TgsSaltDataset)(
            predict_dataset_df,
            is_train=False
        )
        submission_df = delayed(predict)(
            model_paths=top_model_paths,
            log_dir=f"{id}/sub",
            dataset=predict_dataset,
            log_interval=10,
        )

        submission_df = delayed(lambda df: df[['rle_mask']])(submission_df)
        submission_file = delayed(lambda df: df.to_csv(f"{output_dir}/submission.csv"))(
            submission_df,
        )

        self.output = delayed(lambda x: x)((
            top_model_paths,
            scores,
            submission_df,
            submission_file
        ))
