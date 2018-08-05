from cytoolz.curried import keymap, filter, pipe, merge, map
from skimage import io
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import glob


def load_dataset_df(dataset_dir):

    dataset_dir = dataset_dir
    image_dir = os.path.join(dataset_dir, "images")
    mask_dir = os.path.join(dataset_dir, "masks")
    depth_fn = os.path.join(dataset_dir, "depths.csv")
    depth_df = pd.read_csv(
        os.path.join(dataset_dir, "depths.csv")
    )
    depth_df = depth_df.set_index('id')
    df = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
    df = df.drop('rle_mask', axis=1)
    df = df.set_index('id')
    df['image'] = df.index.map(
        lambda x: os.path.join(image_dir, f"{x}.png")
    )
    df['mask'] = df.index.map(
        lambda x: os.path.join(mask_dir, f"{x}.png")
    )
    df = pd.concat([df, depth_df], join='inner', axis=1)
    return df


class TgsSaltDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        depth = self.df['z'].iloc[idx],
        image = torch.DoubleTensor(
            io.imread(self.df['image'].iloc[idx], as_gray=True)
        )
        mask = torch.DoubleTensor(
            io.imread(self.df['mask'].iloc[idx], as_gray=True).astype(bool).astype(float)
        )
        return depth, image, mask
