from cytoolz.curried import keymap, filter, pipe, merge, map
from pathlib import Path
import numpy as np
from dask import delayed


class EarlyStop(object):
    def __init__(self, patience, base_size=1):
        self.patience = patience
        self.losses = []
        self.flag = False
        self.base_size = base_size

    def __call__(self, val_loss):
        self.losses.append(val_loss)
        if len(self.losses) > (self.patience + self.base_size):
            self.losses = self.losses[1:]

        if len(self.losses) == (self.patience + self.base_size):
            self.flag = np.mean(self.losses[:self.base_size]) < np.min(self.losses[self.base_size:])
        return self.flag

    def clear(self):
        self.losses = []

def skip_if_exists(key):
    def _skip(func):
        def wrapper(*args, **kwargs):
            if Path(kwargs[key]).exists():
                return kwargs[key]
            else:
                return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        return wrapper
    return _skip
