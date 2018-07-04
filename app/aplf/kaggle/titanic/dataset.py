from cytoolz.curried import keymap, filter, pipe, merge, map
import torch
from torch.utils.data import Dataset


class TitanicDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,
                 x,
                 y):
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
        return len(self.y)

    def __getitem__(self, idx):
        x = pipe(self.x,
                 map(lambda x: x[idx]),
                 map(torch.FloatTensor),
                 list,
                 torch.cat)
        return x, torch.FloatTensor(self.y[idx])
