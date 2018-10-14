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
        'epochs': 500,
        'labeled_batch_size': 32,
        'no_labeled_batch_size': 8,
        'model_type': 'HUNet',
        'erase_num': 0,
        'erase_p': 0,
        'model_kwargs': {
            'feature_size': 32,
        },
        'consistency': 0.1,
        'center_loss_weight': 0.1,
        'consistency_rampup': 0,

    }
    fine_train_config = {
        'epochs': 400,
        'labeled_batch_size': 32,
        'no_labeled_batch_size': 4,
        'consistency': 0.1,
        'erase_num': 0,
        'erase_p': 0.5,
        'max_factor': 1.0,
        'min_factor': 0.1,
        'period': 5,
        'milestones': [(0, 1)],
        'turning_point': (3, 0.5),
        'lr':0.05,
    }
    g = Graph(
        **base_param,
        id="center-loss",
        base_train_config=base_train_config,
        fine_train_config=fine_train_config,
        n_splits=8,
        top_num=8,
    )

    with Client('dask-scheduler:8786') as c:
        try:
            result = g.output.compute()
            print(result[0])
        finally:
            c.restart()
