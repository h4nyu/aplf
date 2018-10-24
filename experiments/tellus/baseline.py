from sacred import Experiment
from sacred.observers import MongoObserver
from distributed import Client
from aplf.tellus.graph import Graph


EXPERIMENT_ID = "all-loader-1"
Client('dask-scheduler:8786')

ex = Experiment(EXPERIMENT_ID)
ex.observers.append(MongoObserver.create(url='mongo:27017',
                                         db_name='sacred'))


@ex.config
def cfg():
    config = {
        "id": EXPERIMENT_ID,
        "dataset_dir": '/store/tellus',
        "output_dir": '/store/tellus/output/',
        "base_train_config": {
            'epochs': 1000,
            'batch_size': 128,
            'model_type': 'Net',
            'erase_num': 10,
            'erase_p': 0.5,
            'model_kwargs': {
                'feature_size': 64,
            },
            'rgb_loss_weight': 1.5,
            'lr': 0.0001,
        },
        "n_splits": 8,
        "folds": [0],
    }


@ex.automain
def exec_graph(config):
    g = Graph(**config)
    g()
