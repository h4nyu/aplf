from torch.utils.data import Dataset
import torch
from cytoolz.curried import pipe, map, take
import numpy as np
import dask.array as da


class DummyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,
                 start,
                 stop,
                 num,
                 chunks,
                 window_size,
                 annomalies,
                 transform=None):
        self.arry = np.linspace(start, stop, num, chunks, dtype=float)
        self.transform = transform
        self.window_size = window_size
        self.annomalies = annomalies

    def __len__(self):
        return self.arry.shape[0] - self.window_size

    def __getitem__(self, idx):
        label = pipe(self.annomalies,
                     map(lambda x: x[0] <= idx and idx <= x[1]),
                     any)
        ary = self.arry[idx:idx + self.window_size].reshape(1, self.window_size)

        return (torch.tensor(ary, dtype=torch.float), label)
