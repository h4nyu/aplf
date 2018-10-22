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
        'epochs': 400,
        'batch_size': 32,
        'model_type': 'Net',
        'erase_num': 10,
        'erase_p': 0.5,
        'model_kwargs': {
            'feature_size': 64,
        },
        'consistency_loss_wight': 10,
        'center_loss_weight': 0.3,
        'seg_loss_weight': 0.5,
    }

    g = Graph(
        **base_param,
        id="seg-set-5",
        base_train_config=base_train_config,
        n_splits=8,
        top_num=8,
        folds=[0],
    )

    g(scheduler='single-threaded')



