from distributed import Client
from aplf.tellus.graph import Graph
import uuid
from datetime import datetime

base_param = {
}


def test_graph():
    config = {
        "train_config": {
            'epochs': 1000,
            'batch_size': 64,
            'model_kwargs': {
                'feature_size': 8,
                'resize': 64,
                'depth': 3
            },
            'lr': 0.001,
            'neg_scale': 5,
        },
        "dataset_dir": '/store/tellus',
        "output_dir": '/store/tellus/output/split-8-landsatw-100-shuffle-pixel-dropout-0.1-cel-fusion-small-fusion-pad-4-rs-64-landsat-scse-in-res-lr-0.001-esb-1-fs-16-dp-3-4-single-optim-elu-elu',
        'n_splits': 8,
        'folds': [0, 1],
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
