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
        'batch_size': 100,
        'model_type': 'AE',
        'model_kwargs': {
            'feature_size': 128,
            'in_size': (2, 40, 40),
            'out_size': (2, 40, 40),
            'resize': 40,
            'pad': 4
        },
        'rgb_loss_weight': 0.01,
        'pos_loss_weight': 0.1,
        'lr': 0.001,
        'ratio': 100,
    }

    g = Graph(
        **base_param,
        id="ae-elu-0",
        base_train_config=base_train_config,
        n_splits=8,
        folds=[0],
    )

    g(scheduler='single-threaded')


