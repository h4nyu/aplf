from distributed import Client
from aplf.tellus.graph import Graph
import uuid
from datetime import datetime


def test_graph():
    config = {
        "landsat_train_config": {
            'epochs': 400,
            'batch_size': 64,
            'num_ensamble': 1,
            'model_kwargs': {
                'landsat_enc_config':{
                    'feature_size': 8,
                    'depth': 2
                },
                'fusion_enc_config':{
                    'feature_size': 8,
                    'depth': 2
                },
                'resize': 80,
            },
            'landsat_weight': 10,
            'num_ensamble': 1,
            'lr': 0.001,
            'neg_scale': 10,
        },
        "dataset_dir": "/store/tellus",
        "output_dir": "/store/tellus/output/dual-train-ssim-w-2-scse-in-res-lr-0.001-esb-1-fs-8-dp-2-elu",
        "n_splits": 8,
        "folds": list(range(8)),
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
