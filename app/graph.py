from distributed import Client, progress
from joblib import Memory
from dask import visualize
from time import sleep
from cytoolz.curried import pipe, map, take
import torch.nn.functional as F
import dask.bag as db
import numpy as np
import random
import dask.array as da
import torch
import torch.optim as optim
from pprint import pprint
from aplf.utils import merge_kwargs
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from torch.utils.data import DataLoader
from aplf.dataset import DummyDataset
from aplf.model import Autoencoder
from aplf.plot import plot


memory = Memory(cachedir='/', verbose=0)


def gen_dummy():
    return {"loss": np.random.rand(10), 'batch_idx': np.random.rand(10)}


dsk = {
    'dummy_data': (gen_dummy,),
    "plot": (merge_kwargs({'loss': 'z', 'batch_idx': 'x'})(plot), "dummy_data", {
        'from_idx': 0,
        "to_idx": 10000,
        'path': '/data/plot.png'
    }),
}


with Client('dask_scheduler:8786') as c:
    try:
        result = c.get(dsk, 'plot')

    finally:
        c.restart()
