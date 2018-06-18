from distributed import Client
from joblib import Memory
from dask import visualize
from aplf.plot import plot
from aplf.utils import map_kwargs
from torch.utils.data import DataLoader
from aplf.dataset import DummyDataset
import numpy as np


@map_kwargs()
def gen_data(channel, length):
    return np.random.rand(channel, length)


def train(dataset):
    for i in DataLoader(dataset):
        return i


dsk = {
    'dummy_data': (
        gen_data,
        {
            'channel': 10,
            'length': 1000,
        }
    ),
    'dataset': (
        map_kwargs()(DummyDataset),
        'dummy_data',
        {
            'window_size': 10,
        }
    ),
    "train": (
        map_kwargs()(train),
        "dataset"
    ),
    "plot-dataset": (
        map_kwargs({'loss': 'y', 'batch_idx': 'x'})(plot),
        "dataset",
        {
            'path': '/data/plot.png'
        }
    ),
}
visualize(dsk, filename='/data/graph.svg')


with Client('dask_scheduler:8786') as c:
    try:
        result = c.get(dsk, 'dataset')
        print(result)

    finally:
        c.restart()
