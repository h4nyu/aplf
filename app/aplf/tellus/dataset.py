from cytoolz.curried import keymap, filter, pipe, merge, map, compose, concatv
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
import glob
import os
from pathlib import Path
from .preprocess import add_is_empty


def get_row(base_path, sat, label_dir, label):

    after_path = os.path.join(
        base_path,
        sat,
        "after",
        label_dir,
        "*.tif"
    )

    before_path = os.path.join(
        base_path,
        sat,
        "before",
        label_dir,
        "*.tif"
    )

    rows =pipe(
        zip(glob.glob(after_path), glob.glob(before_path)),
        map(lambda x: list(map(Path)(x))),
        map(lambda x: {
            "id": x[0].name,
            "y": label,
            "sat": sat,
            "before_image": x[0],
            "after_image": x[1],
        }),
        list
    )
    return rows


def load_dataset_df(dataset_dir='/store/tellus/train'):
    rows = pipe(
        concatv(
            get_row(
                base_path=dataset_dir,
                sat="LANDSAT",
                label_dir="positive",
                label=True,
            ),
            get_row(
                base_path=dataset_dir,
                sat="LANDSAT",
                label_dir="negative",
                label=False,
            ),
        ),
        list
    )

    df = pd.DataFrame(
        rows
    )
    df = df.set_index('id')
    return df



class TellusDataset(Dataset):
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

