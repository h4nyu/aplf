from distributed import Client
from time import sleep
from cytoolz.curried import pipe, map, take
import dask.bag as db
import numpy as np
import random
import dask.array as da
from pprint import pprint


def load(filename):
    sleep(random.random() * 10)
    return np.fromfile(filename)

def mock(data):
    sleep(random.random() * 10)
    return data



def store(data):
    sleep(random.random() * 10)
    return data.sum().compute()


loads = pipe(range(1, 11),
             map(lambda x: (f"load-{x}", (load, f'/data/batch{x}.dat'))),
             dict)

mocks = pipe(loads,
             map(lambda x: (f"mock-{x}", (mock, x))),
             dict)

dsk = {**loads,
       **mocks,
       'analyze': (da.concatenate, list(mocks.keys())),
       'store': (store, 'analyze')}
pprint(dsk)


client = Client('dask_scheduler:8786')
result = client.get(dsk, ['store'])  # executes in parallel
print(result)
