from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk
from sklearn.model_selection import train_test_split
from dask import delayed
from datetime import datetime
import numpy as np
import pandas as pd
import uuid
from aplf import config
from .dataset import TgsSaltDataset, load_dataset_df
from .train import train
from .predict import predict
from .preprocess import take_topk, cleanup, cut_bin, add_cover, groupby, avarage_dfs, dump_json, divide_by_cover


class Graph(object):
    def __init__(self,
                 id,
                 dataset_dir,
                 output_dir,
                 epochs,
                 model_type,
                 val_split_size,
                 labeled_batch_size,
                 no_labeled_batch_size,
                 patience,
                 base_size,
                 parallel,
                 top_num,
                 feature_size,
                 ema_decay,
                 consistency,
                 consistency_rampup,
                 depth,
                 cyclic_period,
                 switch_epoch,
                 milestones,
                 ):
        params = locals()

        ids = pipe(
            range(parallel),
            map(lambda x: uuid.uuid4()),
            list
        )

        dataset_df = delayed(load_dataset_df)(
            dataset_dir,
            'train.csv'
        )
        dataset_df = delayed(cleanup)(dataset_df)

        spliteds = pipe(
            ids,
            enumerate,
            map(lambda x: delayed(train_test_split)(
                dataset_df,
                test_size=val_split_size,
                shuffle=True,
                random_state=x[0]
            )),
            list
        )
        train_datasets = pipe(
            spliteds,
            map(delayed(lambda x: x[0])),
            map(lambda x: delayed(TgsSaltDataset)(
                x,
                has_y=True
            )),
            list
        )

        val_datasets = pipe(
            spliteds,
            map(delayed(lambda x: x[1])),
            map(lambda x: delayed(TgsSaltDataset)(x, has_y=True)),
            list
        )

        predict_dataset_df = delayed(load_dataset_df)(
            dataset_dir,
            'sample_submission.csv'
        )

        predict_dataset = delayed(TgsSaltDataset)(
            predict_dataset_df,
            has_y=False
        )

        model_paths = pipe(
            zip(ids, train_datasets, val_datasets),
            map(lambda x: delayed(train)(
                model_path=f"{output_dir}/model_{x[0]}.pt",
                train_dataset=x[1],
                val_dataset=x[2],
                model_type=model_type,
                no_labeled_dataset=predict_dataset,
                epochs=epochs,
                labeled_batch_size=labeled_batch_size,
                no_labeled_batch_size=no_labeled_batch_size,
                feature_size=feature_size,
                patience=patience,
                base_size=base_size,
                consistency=consistency,
                consistency_rampup=consistency_rampup,
                ema_decay=ema_decay,
                depth=depth,
                cyclic_period=cyclic_period,
                switch_epoch=switch_epoch,
                milestones=milestones,
                log_dir=f'{config["TENSORBORAD_LOG_DIR"]}/{id}/{x[0]}',
            )),
            list
        )

        scores = pipe(
            zip(ids, model_paths, val_datasets),
            map(lambda x: delayed(predict)(
                model_paths=[x[1]],
                log_dir=f'{config["TENSORBORAD_LOG_DIR"]}/{id}/{x[0]}/val',
                dataset=x[2],
                log_interval=1,
            )),
            map(delayed(lambda df: df['score'].mean())),
            list
        )

        param_files = pipe(
            zip(ids, scores),
            map(lambda x: dump_json(
                f"{output_dir}/{x[0]}.json",
                {
                    'graph_id': id,
                    'model_id': x[0],
                    'score': x[1],
                    'params': params
                },
            )),
            list
        )

        top_model_paths = delayed(take_topk)(scores, model_paths, top_num)

        submission_df = delayed(predict)(
            model_paths=top_model_paths,
            log_dir=f'{config["TENSORBORAD_LOG_DIR"]}/{id}/sub',
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
            submission_file,
            param_files,
        ))


class SelectionGraph(object):
    def __init__(self,
                 id,
                 random_state,
                 dataset_dir,
                 output_dir,
                 val_split_size,
                 n_generation,
                 parallel,
                 bins,
                 train_config,
                 ):
        ids = pipe(
            range(parallel),
            map(lambda x: uuid.uuid4()),
            list
        )

        dataset_df = delayed(load_dataset_df)(
            dataset_dir,
            'train.csv'
        )
        dataset_df = delayed(cleanup)(dataset_df)

        splited = delayed(train_test_split)(
            dataset_df,
            test_size=val_split_size,
            shuffle=True,
            random_state=random_state
        )
        train_df = delayed(lambda x: x[0])(splited)
        train_df = delayed(add_cover)(train_df)
        train_dfs = delayed(divide_by_cover)(train_df, bins=4)

        val_df = delayed(lambda x: x[0])(splited)

        predict_dataset_df = delayed(load_dataset_df)(
            dataset_dir,
            'sample_submission.csv'
        )

        before_model_path=None
        for n in range(n_generation):
            traineds = pipe(
                zip(ids),
                map(lambda x: delayed(train)(
                    **train_config,
                    before_model_path=before_model_path,
                    model_path=f"{output_dir}/model_{x[0]}.pt",
                    train_dfs=train_dfs,
                    val_df=val_df,
                    no_labeled_df=predict_dataset_df,
                    log_dir=f'{config["TENSORBORAD_LOG_DIR"]}/{id}/{x[0]}',
                )),
                list
            )
            before_model_path = delayed(lambda x: pipe(
                topk(1, x, key=lambda y: y['score']),
                first,
                lambda x: x['model_path']
            ))(traineds)

        #  predicted = delayed(predict)(
        #      model_paths=[before_model_path],
        #      log_dir=f'{config["TENSORBORAD_LOG_DIR"]}/{id}/val',
        #      dataset_df=val_df,
        #      log_interval=1,
        #  )
        #  score = delayed(lambda df: df['score'].mean())(predicted)
        #
        self.output = delayed(lambda x: x)((
            #  score,
            #  train_dfs,
            before_model_path,
        ))
