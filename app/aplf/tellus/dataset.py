from cytoolz.curried import keymap, filter, pipe, merge, map, compose, concatv, first
import random
from torch.utils.data import Subset, Sampler
from skimage import img_as_float
from dask import delayed
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
import h5py
from .preprocess import add_is_empty
from aplf.utils import skip_if_exists


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

    rows = pipe(
        zip(glob.glob(after_path), glob.glob(before_path)),
        map(lambda x: list(map(Path)(x))),
        map(lambda x: {
            "id": x[0].name,
            "label": label,
            "sat": sat,
            "before_image": str(x[0]),
            "after_image": str(x[1]),
        }),
        list
    )
    return rows


@skip_if_exists("output")
def load_dataset_df(dataset_dir='/store/tellus/train',
                    output='/store/tellus/train.pqt'):
    rows = pipe(
        concatv(
            get_row(
                base_path=dataset_dir,
                sat="PALSAR",
                label_dir="positive",
                label=1,
            ),
            get_row(
                base_path=dataset_dir,
                sat="PALSAR",
                label_dir="negative",
                label=0,
            ),
        ),
        list
    )
    df = pd.DataFrame(rows)
    df.set_index('id')
    df.to_parquet(output, compression='gzip')
    return output


def read_image(path):
    return io.imread(
        path,
        as_gray=True
    )


def image_to_tensor(path):
    image = io.imread(
        path,
        as_gray=True
    )
    image = img_as_float(image)
    h, w = image.shape
    image = image.reshape(1, h, w)
    tensor = torch.FloatTensor(image)
    return tensor


class TellusDataset(Dataset):
    def __init__(self, df, has_y=True):
        self.has_y = has_y
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id = self.df.index[idx]
        row = self.df.iloc[idx]
        before = image_to_tensor(
            row['before_image']
        )

        after = image_to_tensor(
            row['after_image']
        )

        if self.has_y:
            return {
                'id': id,
                'before': before,
                'after': after,
                'label': row['label'],
            }
        else:
            return {
                'id': id,
                'before': transform(before),
                'after': transform(after),
            }


def to_pn_set(subset):
    df = subset.dataset.df
    filtered_df = df.ix[subset.indices]
    pos_indices = filtered_df[filtered_df['label'] == 1].index
    neg_indices = filtered_df[filtered_df['label'] == 0].index
    pos_set = Subset(
        dataset=subset.dataset,
        indices=pos_indices
    )

    neg_set = Subset(
        dataset=subset.dataset,
        indices=neg_indices
    )
    return pos_set, neg_set


class ChunkSampler(Sampler):
    def __init__(self,  epoch_size, len_indices, shuffle=True):
        self.shuffle = shuffle
        self.epoch_size = epoch_size
        self.len_indices = len_indices
        indices = range(len_indices)
        self.chunks = pipe(
            range(0, len_indices//epoch_size),
            map(lambda x: indices[x*epoch_size:(x+1)*epoch_size]),
            map(list),
            list,
        )
        self.chunk_idx = 0

    def __iter__(self):
        chunk = self.chunks[self.chunk_idx]
        if self.shuffle:
            random.shuffle(chunk)
        for i in chunk:
            yield i
        self.chunk_idx += 1
        if self.chunk_idx >= len(self.chunks):
            self.chunk_idx = 0

    def __len__(self):
        return self.epoch_size
