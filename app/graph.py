from distributed import Client
from time import sleep
import random


def load(filename):
    sleep(random.random() / 10)
    pass


def clean(data):
    sleep(random.random() / 10)
    pass


def analyze(sequence_of_data):
    sleep(random.random() / 10)
    pass


def store(result):
    sleep(random.random() / 10)
    pass


dsk = {'load-1': (load, 'myfile.a.data'),
       'load-2': (load, 'myfile.b.data'),
       'load-3': (load, 'myfile.c.data'),
       'clean-1': (clean, 'load-1'),
       'clean-2': (clean, 'load-2'),
       'clean-3': (clean, 'load-3'),
       'analyze': (analyze, ['clean-%d' % i for i in [1, 2, 3]]),
       'store': (store, 'analyze')}


client = Client('dask_scheduler:8786')
client.get(dsk, 'store')  # executes in parallel
