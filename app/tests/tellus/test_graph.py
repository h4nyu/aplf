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
        'batch_size': 256,
        'model_type': 'MultiEncoder',
        'model_kwargs': {
            'feature_size': 32,
            'resize': 40,
            'pad': 4,
            'depth': 3
        },
        'rgb_loss_weight': 0.001,
        'lr': 0.0001,
    }

    g = Graph(
        **base_param,
        id="multi-0",
        train_method='multi',
        base_train_config=base_train_config,
        n_splits=8,
        folds=[0],
    )

    g(scheduler='single-threaded')
