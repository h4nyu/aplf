from cytoolz.curried import keymap, filter, pipe, merge, map
import torch
from torch.utils.data import Dataset
import numpy as np


class TitanicDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,
                 x,
                 y=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x[0])

    def __getitem__(self, idx):
        x = pipe(self.x,
                 map(lambda x: x[idx]),
                 map(torch.FloatTensor),
                 list,
                 torch.cat)
        if self.y is None:
            return (x,)
        else:
            return (x, torch.FloatTensor(self.y[idx]))

