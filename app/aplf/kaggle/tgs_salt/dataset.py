from cytoolz.curried import keymap, filter, pipe, merge, map
from skimage import io
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import glob


class TgsSaltDataset(Dataset):
    def __init__(self,
                 dataset_dir):
        self.dataset_dir = dataset_dir
        self.image_dir = os.path.join(dataset_dir, "images")
        self.mask_dir = os.path.join(dataset_dir, "masks")
        self.depth_fn = os.path.join(dataset_dir, "depths.csv")
        depth_df = pd.read_csv(
            os.path.join(self.dataset_dir, "depths.csv")
        )
        depth_df = depth_df.set_index('id')
        df = pd.read_csv(os.path.join(self.dataset_dir, "train.csv"))
        df = df.drop('rle_mask', axis=1)
        df = df.set_index('id')
        df['image'] = df.index.map(
            lambda x: os.path.join(self.image_dir, f"{x}.png"))
        df['mask'] = df.index.map(
            lambda x: os.path.join(self.mask_dir, f"{x}.png"))
        self.df = pd.concat([df, depth_df], join='inner', axis=1)
        self.x = self.df[['z', 'image']]
        self.y = self.df['mask']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = (
            io.imread(self.df['image'][idx], as_gray=True),
            self.df['z'][idx]
        )
        y = io.imread(self.df['mask'][idx], as_gray=True),
        return x, y
