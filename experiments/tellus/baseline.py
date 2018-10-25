from sacred import Experiment
from sacred.observers import MongoObserver
from distributed import Client
from aplf.tellus.graph import Graph
import os
print()
print()


EXPERIMENT_ID = "all-loader-resize-rgb-weight-0.5"
MASTER_IP = os.environ['MASTER_IP']
MONGO_PORT = os.environ['MONGO_PORT']

ex = Experiment(EXPERIMENT_ID)
ex.observers.append(MongoObserver.create(
    url=f'{MASTER_IP}:{MONGO_PORT}',
    db_name='sacred')
)


@ex.config
def cfg():
    config = {
        "id": EXPERIMENT_ID,
        "dataset_dir": '/store/tellus',
        "output_dir": '/store/tellus/output/',
        "base_train_config": {
            'epochs': 1000,
            'batch_size': 32,
            'model_type': 'Net',
            'erase_num': 10,
            'erase_p': 0.5,
            'model_kwargs': {
                'feature_size': 64,
                'resize': 120,
                'pad': 4
            },
            'rgb_loss_weight': 0.5,
            'lr': 0.001,
        },
        "n_splits": 8,
        "folds": [0],
    }


@ex.automain
def exec_graph(config):
    g = Graph(**config)
    g(scheduler='single-threaded')
