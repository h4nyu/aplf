from dask.multiprocessing import get


def load(*args):
    print('load')
    print(args)


def clean(*args):
    print('clean')
    print(args)
    pass


def analyze(*args):
    print('analyze')
    print(args)
    pass


def store(*args):
    print('store')
    print(args)


dsk = {'load-1': (load, 'myfile.a.data'),
       'load-2': (load, 'myfile.b.data'),
       'load-3': (load, 'myfile.c.data'),
       'clean-1': (clean, 'load-1'),
       'clean-2': (clean, 'load-2'),
       'clean-3': (clean, 'load-3'),
       'analyze': (analyze, ['clean-%d' % i for i in [1, 2, 3]]),
       'store': (store, 'analyze')}
print(get(dsk, 'store'))  # executes in parallel

import dask.array as da
x = da.ones((15, 15), chunks=(5, 5))

y = x + x.T

# y.compute()
y.visualize(filename='transpose.svg')

