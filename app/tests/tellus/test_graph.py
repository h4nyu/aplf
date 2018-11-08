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
        'batch_size': 64,
        'model_type': 'MultiEncoder',
        'num_ensamble': 2,
        'model_kwargs': {
            'feature_size': 16,
            'resize': 80,
            'pad': 4,
            'depth': 2
        },
        'landsat_weight': 10,
        'num_ensamble': 3,
        'lr': 0.001,
        'neg_scale': 10,
    }

    g = Graph(
        **base_param,
        id=f"3795fe71-11b5-46a8-9e82-769d5a164f33-scse-in-res-lr-0.001-esb-2-fs-16-dp2-lw-10",
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
