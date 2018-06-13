
from distributed import Client
from time import sleep
import random


def inc(x):
    sleep(random.random() / 10)
    return x + 1


def dec(x):
    sleep(random.random() / 10)
    return x - 1


def add(x, y):
    sleep(random.random() / 10)
    return x + y


client = Client('dask_scheduler:8786')  # set up local cluster on your laptop
incs = client.map(inc, range(10000))
decs = client.map(dec, range(10000))
adds = client.map(add, incs, decs)
total = client.submit(sum, adds)

del incs, decs, adds
print(total.result())
