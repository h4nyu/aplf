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
        'model_type': 'Net',
        'erase_num': 10,
        'erase_p': 0.5,
        'model_kwargs': {
            'feature_size': 64,
        },
        'consistency_loss_wight': 10,
        'center_loss_weight': 0.3,
        'rgb_loss_weight': 1,
        'lr': 0.001,
    }

    g = Graph(
        **base_param,
        id="lovas-1",
        base_train_config=base_train_config,
        n_splits=8,
        top_num=8,
        folds=[0],
    )

    g(scheduler='single-threaded')

