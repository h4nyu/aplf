from cytoolz.curried import keymap, filter, pipe, merge, map
import torch
from torch.utils.data import Dataset
import numpy as np


class TitanicDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,
                 df,
                 y_column='Survived'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.y_column = y_column
        self.x_columns = df.columns != y_column
        self.x_df = self.df.loc[:, self.x_columns]
        self.y_df = self.df[y_column]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(np.concatenate(self.x_df.iloc[idx], axis=0)),
            self.y_df[idx]
        )
