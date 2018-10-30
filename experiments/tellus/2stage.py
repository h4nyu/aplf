from sacred import Experiment
from sacred.observers import MongoObserver
from distributed import Client
from aplf.tellus.multi_stage import Graph
import os


EXPERIMENT_ID = "multi-stage-0"
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
            'batch_size': 128,
            'model_type': 'AE',
            'model_kwargs': {
                'feature_size': 64,
                'depth': 3,
                'resize': 80,
                'pad': 10
            },
            'lr': 0.001,
        },
        "n_splits": 8,
        "folds": [0],
    }


@ex.automain
def exec_graph(config):
    g = Graph(**config)
    g(scheduler='single-threaded')
