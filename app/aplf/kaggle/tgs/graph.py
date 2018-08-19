from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk
from sklearn.model_selection import train_test_split
from dask import delayed
import numpy as np
from .dataset import TgsSaltDataset, load_dataset_df
from .train import train, boost_fit
from .predict import predict
from .preprocess import take_topk


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

        model_paths = []
        for model_id, (train_dataset, val_dataset) in enumerate(zip(train_datasets, val_datasets)):
            fitted = delayed(boost_fit)(
                prev_model_paths=model_paths,
                model_id=model_id,
                model_path=f"{output_dir}/model_{model_id}.pt",
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                base_size=base_size,
            )
            model_paths.append(fitted)

        scores = pipe(val_datasets,
                      map(lambda x: delayed(predict)(
                          model_paths=model_paths,
                          output_dir=f"predict/val",
                          dataset=x,
                      )),
                      map(delayed(lambda df: df['score'].mean())),
                      list)


        submission_df = delayed(load_dataset_df)(
            dataset_dir,
            'sample_submission.csv'
        )

        submission_dataset = delayed(TgsSaltDataset)(
            submission_df,
            is_train=False
        )

        submission_df = delayed(predict)(
            model_paths=model_paths,
            output_dir=f"predict/sub",
            dataset=submission_dataset,
        )

        submission_df = delayed(lambda df: df[['rle_mask']])(submission_df)
        submission_file = delayed(lambda df: df.to_csv(f"{output_dir}/submission.csv"))(
            submission_df,
        )

        self.output = delayed(lambda x: x)((
            scores,
            submission_file
        ))
