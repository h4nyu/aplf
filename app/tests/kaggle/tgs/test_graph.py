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
        'batch_size': 32,
        'model_type': 'HUNet',
        'model_kwargs': {
            'feature_size': 32,
            'depth': 3,
        },
        'erase_num': 5,

    }
    fine_train_config = {
        'epochs': 200,
        'labeled_batch_size': 32,
        'no_labeled_batch_size': 4,
        'consistency': 1,
        'erase_num': 3,
    }
    g = Graph(
        **base_param,
        id="18",
        base_train_config=base_train_config,
        fine_train_config=fine_train_config,
        n_splits=4,
        top_num=2,
    )

    with Client('dask-scheduler:8786') as c:
        try:
            result = g.output.compute()
            print(result[0])
        finally:
            c.restart()
