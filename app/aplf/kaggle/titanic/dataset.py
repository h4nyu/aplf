from cytoolz.curried import keymap, filter, pipe, merge, map
import torch
from torch.utils.data import Dataset
import numpy as np


class TitanicDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,
                 x_df,
                 y_df=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.x_df = x_df
        self.y_df = y_df

    def __len__(self):
        return len(self.x_df)

    def __getitem__(self, idx):
        if self.y_df is not None:
            return (
                torch.FloatTensor(np.concatenate(self.x_df.iloc[idx], axis=0)),
                self.y_df[idx]
            )
        else:
            return torch.FloatTensor(np.concatenate(self.x_df.iloc[idx], axis=0)),
