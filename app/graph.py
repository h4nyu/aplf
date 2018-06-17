from distributed import Client
from joblib import Memory
from aplf.plot import plot
from aplf.utils import merge_kwargs
import numpy as np


memory = Memory(cachedir='/', verbose=0)


def gen_dummy():
    return {"loss": np.random.rand(10), 'batch_idx': np.random.rand(10)}


dsk = {
    'dummy_data': (gen_dummy,),
    "plot": (merge_kwargs({'loss': 'y', 'batch_idx': 'x'})(plot), "dummy_data", {
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
