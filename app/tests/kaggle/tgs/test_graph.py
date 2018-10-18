from distributed import Client
from aplf.kaggle.tgs.graph import Graph
import uuid
from datetime import datetime

base_param = {
    "dataset_dir": '/store/kaggle/tgs',
    "output_dir": '/store/kaggle/tgs/output/',
}


def test_graph():
    base_train_config = {
        'epochs': 400,
        'batch_size': 48,
        'model_type': 'HUNet',
        'erase_num': 5,
        'erase_p': 1,
        'model_kwargs': {
            'feature_size': 32,
        },
        'consistency_loss_wight': 2,
        'center_loss_weight': 0.1,
        'seg_loss_weight': 0.5,

    }
    fine_train_config = {
        'epochs': 400,
        'batch_size': 32,
        'no_label_batch_size': 4,
        'erase_num': 5,
        'erase_p': 0.5,
        'consistency_loss_wight': 5,
        'center_loss_weight': 0.2,
        'seg_loss_weight': 0.5,
        'scheduler_config':{
            'max_factor': 2.0,
            'min_factor': 0.5,
            'period': 5,
            'milestones':[(0, 1.0)]
        }
    }
    g = Graph(
        **base_param,
        id="seg-set-sub",
        base_train_config=base_train_config,
        fine_train_config=fine_train_config,
        n_splits=8,
        top_num=8,
        #  folds=[0, 1, 3, 7],
        folds=[4],
    )

    with Client('dask-scheduler:8786') as c:
        try:
            result = g.output.compute()
            print(result[0])
        finally:
            c.restart()
