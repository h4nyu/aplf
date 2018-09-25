from cytoolz.curried import keymap, filter, pipe, merge, map, compose
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
from .preprocess import hflip, vflip, crop


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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id = self.df.index[idx]
        depth = self.df['z'].iloc[idx],
        img_ary = io.imread(
            self.df['x_image'].iloc[idx],
            as_gray=True
        )
        h, w = img_ary.shape
        limit_h = h // 5
        limit_w = w // 5
        start = (
            np.random.randint(0, limit_w),
            np.random.randint(0, limit_h),
        )

        end = (
            np.random.randint(w - limit_w, w),
            np.random.randint(h - limit_h, h),
        )

        transforms = [
            torch.FloatTensor,
            lambda x: x.view(1, h, w),
        ]

        if self.has_y:
            mask_ary = io.imread(
                self.df['y_mask_true'].iloc[idx],
                as_gray=True
            ).astype(bool).astype(int)

            transform = compose(*reversed([
                *transforms,
                #  random.choice([
                #      lambda x: x,
                #      crop(start=start, end=end)
                #  ]),
                random.choice([
                    lambda x: x,
                    hflip,
                ]),
            ]))

            return {
                'id': id,
                'depth': depth,
                'image': transform(img_ary),
                'mask': transform(mask_ary),
            }
        else:
            transform = compose(*reversed([
                *transforms
            ]))
            return {
                'id': id,
                'depth': depth,
                'image': transform(img_ary),
            }
