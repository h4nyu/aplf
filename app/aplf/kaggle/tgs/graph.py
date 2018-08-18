from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk
from sklearn.model_selection import train_test_split
from dask import delayed
import numpy as np
from .dataset import TgsSaltDataset, load_dataset_df
from .train import train
from .predict import predict


class Graph(object):
    def __init__(self,
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

        ids = list(range(parallel))

        dataset_df = delayed(load_dataset_df)(dataset_dir, 'train.csv')
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
            map(delayed(TgsSaltDataset)),
            list
        )

        val_datasets = pipe(
            spliteds,
            map(delayed(lambda x: x[1])),
            map(delayed(TgsSaltDataset)),
            list
        )

        model_paths = pipe(
            zip(train_datasets, val_datasets),
            enumerate,
            map(lambda x: delayed(train)(
                model_id=x[0],
                model_path=f"{output_dir}/model_{x[0]}.pt",
                train_dataset=x[1][0],
                val_dataset=x[1][1],
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                base_size=base_size,
            )),
            list
        )

        scores = pipe(zip(val_datasets, model_paths),
                      map(lambda x: delayed(predict)(
                          model_paths=[x[1]],
                          output_dir=f"predict/val",
                          dataset=x[0],
                          log_interval=1,
                      )),
                      map(delayed(lambda df: df['score'].mean())),
                      list)

        top_model_paths = delayed(lambda s, p: list(topk(top_num, zip(s, p), key=lambda x: x[0])))(
            scores,
            model_paths
        )

        submission_df = delayed(load_dataset_df)(
            dataset_dir,
            'sample_submission.csv'
        )

        submission_dataset = delayed(TgsSaltDataset)(
            submission_df,
            is_train=False
        )

        submission_df = delayed(predict)(
            model_paths=top_model_paths,
            output_dir=f"predict/sub",
            dataset=submission_dataset,
            log_interval=100,
        )

        submission_df = delayed(lambda df: df[['rle_mask']])(submission_df)
        submission_file = delayed(lambda df: df.to_csv(f"{output_dir}/submission.csv"))(
            submission_df,
        )

        self.output = delayed(lambda x: x)((
            top_model_paths,
            scores,
            submission_file
        ))
