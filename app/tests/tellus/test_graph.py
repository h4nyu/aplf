from distributed import Client
from aplf.tellus.graph import Graph
import uuid
from datetime import datetime

base_param = {
    "dataset_dir": '/store/tellus',
    "output_dir": '/store/tellus/output/',
}


def test_graph():
    base_train_config = {
        'epochs': 1000,
        'batch_size': 128,
        'val_batch_size': 512,
        'model_type': 'MultiEncoder',
        'num_ensamble': 2,
        'model_kwargs': {
            'feature_size': 8,
            'resize': 120,
            'pad': 4,
            'depth': 2
        },
        'divides': 10,
        'landsat_weight': 0.1,
        'num_ensamble': 2,
        'lr': 0.001,
    }

    g = Graph(
        **base_param,
        id="repo-1",
        train_method='multi',
        base_train_config=base_train_config,
        n_splits=10,
        folds=[0],
    )


    with Client('dask-scheduler:8786') as c:
        try:
            result = g()
        finally:
            c.restart()
