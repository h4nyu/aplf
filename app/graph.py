from dask.multiprocessing import get
from dask import visualize


def load(filename):
    pass


def clean(data):
    pass


def analyze(sequence_of_data):
    pass


def store(result):
    pass


dsk = {'load-1': (load, 'myfile.a.data'),
       'load-2': (load, 'myfile.b.data'),
       'load-3': (load, 'myfile.c.data'),
       'clean-1': (clean, 'load-1'),
       'clean-2': (clean, 'load-2'),
       'clean-3': (clean, 'load-3'),
       'analyze': (analyze, ['clean-%d' % i for i in [1, 2, 3]]),
       'store': (store, 'analyze')}


visualize(dsk, '/data/graph.svg')
get(dsk, 'store')  # executes in parallel
