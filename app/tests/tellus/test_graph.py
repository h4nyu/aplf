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
        'epochs': 300,
        'batch_size': 516,
        'model_type': 'MultiEncoder',
        'num_ensamble': 1,
        'model_kwargs': {
            'feature_size': 8,
            'resize': 80,
            'depth': 2
        },
        'landsat_weight': 2,
        'num_ensamble': 1,
        'lr': 0.01,
    }

    g = Graph(
        **base_param,
        id=f"acc-batch-scse-in-res-lr-0.001-esb-1-fs-8-dp-2-dual-optim-elu",
        train_method='multi',
        base_train_config=base_train_config,
        n_splits=8,
        folds=[0],
    )

    g(scheduler='single-threaded')
    #
    #  with Client('dask-scheduler:8786') as c:
    #      try:
    #          result = g()
    #      finally:
    #          c.restart()
    #
