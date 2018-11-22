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
            'batch_size': 64,
            'model_kwargs': {
                'feature_size': 8,
                'resize': 64,
                'depth': 3
            },
            'lr': 0.001,
            'neg_scale': 10,
            'fusion_train_start': 100,
        },
        "dataset_dir": '/store/tellus',
        "output_dir": '/store/tellus/output/repro-3-fusion-start-100-th-large-fusion-padr-10-rs-32-landsat-scse-in-res-lr-0.001-esb-1-fs-8-dp-2-dual-optim-relu',
        'n_splits': 15,
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
