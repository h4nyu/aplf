from cytoolz.curried import keymap, filter, pipe, merge, map, compose
from torchvision.transforms import (
    RandomRotation,
    ToPILImage,
    Compose, ToTensor,
    CenterCrop,
    RandomAffine,
    TenCrop,
    RandomApply,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomResizedCrop,
)
import random
from skimage import io
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import glob
import random
import torch.nn.functional as F
from torchvision.transforms.functional import hflip, vflip, rotate


def load_dataset_df(dataset_dir, csv_fn='train.csv'):

    dataset_dir = dataset_dir
    image_dir = os.path.join(dataset_dir, "images")
    mask_dir = os.path.join(dataset_dir, "masks")
    depth_fn = os.path.join(dataset_dir, "depths.csv")
    depth_df = pd.read_csv(
        os.path.join(dataset_dir, "depths.csv")
    )
    depth_df = depth_df.set_index('id')
    df = pd.read_csv(os.path.join(dataset_dir, csv_fn))
    df = df.set_index('id')
    df['x_image'] = df.index.map(
        lambda x: os.path.join(image_dir, f"{x}.png")
    )
    df['y_mask_true'] = df.index.map(
        lambda x: os.path.join(mask_dir, f"{x}.png")
    )
    df = pd.concat([df, depth_df], join='inner', axis=1)
    return df


class TgsSaltDataset(Dataset):
    def __init__(self, df, has_y=True):
        self.has_y = has_y
        self.df = df
        self.transforms = [
            lambda x:x,
            hflip,
        ]


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id = self.df.index[idx]
        depth = self.df['z'].iloc[idx],
        image = torch.FloatTensor(
            io.imread(
                self.df['x_image'].iloc[idx],
                as_gray=True
            ).reshape(1, 101, 101)
        )
        if self.has_y:
            transform = Compose([
                ToPILImage(),
                random.choice(self.transforms),
                random.choice(self.transforms),
                ToTensor()
            ])
            mask = torch.FloatTensor(
                io.imread(
                    self.df['y_mask_true'].iloc[idx],
                    as_gray=True
                ).astype(bool).astype(float).reshape(1, 101, 101)
            )
            return {
                'id': id,
                'depth': depth,
                'image': transform(image),
                'mask': transform(mask),
            }
        else:
            return {
                'id': id,
                'depth': depth,
                'image': image,
            }

