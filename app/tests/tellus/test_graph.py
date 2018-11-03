from distributed import Client
from aplf.tellus.graph import Graph
import uuid
from datetime import datetime
import uuid

base_param = {
    "dataset_dir": '/store/tellus',
    "output_dir": '/store/tellus/output/',
}


def test_graph():
    base_train_config = {
        'epochs': 1000,
        'batch_size': 256,
        'val_batch_size': 516,
        'model_type': 'MultiEncoder',
        'num_ensamble': 2,
        'validate_interval': 10,
        'model_kwargs': {
            'feature_size': 16,
            'resize': 80,
            'pad': 4,
            'depth': 1
        },
        'landsat_weight': 0.5,
        'num_ensamble': 2,
        'lr': 0.001,
    }

    g = Graph(
        **base_param,
        id=f"{uuid.uuid4()}_bag_1_lr_1e-3_cbam_fs_16",
        train_method='multi',
        base_train_config=base_train_config,
        n_splits=10,
        folds=[0],
    )

    g(scheduler='single-threaded')
