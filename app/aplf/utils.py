from cytoolz.curried import keymap, filter, pipe, merge, map
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
