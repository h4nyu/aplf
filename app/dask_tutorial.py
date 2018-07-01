from distributed import Client
from time import sleep
import random


def inc(x):
    sleep(random.random() )
    return x + 1


def dec(x):
    sleep(random.random() )
    return x - 1


def add(x, y):
    sleep(random.random() )
    return x + y


client = Client('dask-scheduler:8786')

incs = client.map(inc, range(100))
decs = client.map(dec, range(100))
adds = client.map(add, incs, decs)
total = client.submit(sum, adds)

del incs, decs, adds
total.result()
