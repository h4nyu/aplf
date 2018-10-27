from sacred import Experiment
from sacred.observers import MongoObserver
from distributed import Client
from aplf.tellus.graph import Graph
import os


EXPERIMENT_ID = "ae-4"
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
        'train_method': 'ae',
        "base_train_config": {
            'epochs': 1000,
            'batch_size': 256,
            'model_type': 'AE',
            'model_kwargs': {
                'feature_size': 64,
                'in_size': (2, 40, 40),
                'out_size': (2, 40, 40),
                'resize': 40,
                'pad': 4
            },
            'rgb_loss_weight': 0.01,
            'pos_loss_weight': 0.01,
            'lr': 0.001,
            'ratio': 10,
        },
        "n_splits": 8,
        "folds": [0],
    }


@ex.automain
def exec_graph(config):
    g = Graph(**config)
    g(scheduler='single-threaded')
