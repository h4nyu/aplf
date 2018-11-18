from distributed import Client
from aplf.tellus.graph import Graph
import uuid
from datetime import datetime


def test_graph():
    model_kwargs = {
        'landsat_enc_config': {
            'feature_size': 16,
            'depth': 2
        },
        'fusion_enc_config': {
            'feature_size': 8,
            'depth': 2
        },
        'resize': 80,
    }
    config = {
        "landsat_train_config": {
            'epochs': 400,
            'batch_size': 32,
            'model_kwargs': model_kwargs,
            'landsat_weight': 10,
            'lr': 0.001,
            'neg_scale': 10,
        },
        "palsar_train_config": {
            'epochs': 400,
            'batch_size': 32,
            'model_kwargs': model_kwargs,
            'lr': 0.001,
            'neg_scale': 10,
        },
        "dataset_dir": "/store/tellus",
        "output_dir": "/store/tellus/output/dual-optim-ssim-w-2-scse-in-res-lr-0.001-fs-16-8-dp-2-2-elu",
        "n_splits": 8,
        "folds": [0],
        'seed': 0

    }
    g = Graph(config)

    g().compute(scheduler='single-threaded')
