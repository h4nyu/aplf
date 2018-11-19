from distributed import Client
from aplf.tellus.graph import Graph
import uuid
from datetime import datetime

base_param = {
}


def test_graph():
    config = {
        "train_config": {
            'epochs': 400,
            'batch_size': 128,
            'model_kwargs': {
                'feature_size': 8,
                'resize': 80,
                'depth': 2
            },
            'landsat_weight': 0.5,
            'lr': 0.001,
            'neg_scale': 10,
        },
        "dataset_dir": '/store/tellus',
        "output_dir": '/store/tellus/output/direct-landsat-scse-in-res-lr-0.001-esb-1-fs-8-dp-2-dual-optim-elu',
        'n_splits': 8,
        'folds': [0],
        'seed': 0
    }
    g = Graph(config)

    g().compute(scheduler='single-threaded')
    #
    #  with Client('dask-scheduler:8786') as c:
    #      try:
    #          result = g()
    #      finally:
    #          c.restart()
    #
