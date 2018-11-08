from cytoolz.curried import keymap, filter, pipe, merge, map, compose, concatv, first, valmap, curry
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
    RandomCrop,
)

from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_gamma
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
from sklearn.model_selection import KFold


def get_train_row(base_path, label_dir, label):
    rows = pipe(
        [
            ("PALSAR", "before"),
            ("PALSAR",  "after"),
            ("LANDSAT", "before"),
            ("LANDSAT",  "after"),
        ],
        map(lambda x: (base_path, *x, label_dir, "*.tif")),
        map(lambda x: os.path.join(*x)),
        map(glob.glob),
        list,
        lambda x: zip(*x),
        map(lambda x: list(map(Path)(x))),
        map(lambda x: {
            "id": x[0].name,
            "label": label,
            "palsar_before": str(x[0]),
            "palsar_after": str(x[1]),
            "landsat_before": str(x[2]),
            "landsat_after": str(x[3]),
        }),
        list
    )
    return rows


def get_test_row(base_path):
    rows = pipe(
        [
            ("PALSAR", "before"),
            ("PALSAR", "after"),
        ],
        map(lambda x: (base_path, *x, "*.tif")),
        map(lambda x: os.path.join(*x)),
        map(glob.glob),
        list,
        lambda x: zip(*x),
        map(lambda x: list(map(Path)(x))),
        map(lambda x: {
            "id": str(x[0].name),
            "palsar_before": str(x[0]),
            "palsar_after": str(x[1]),
        }),
        list
    )
    return rows


@skip_if_exists("output")
def load_train_df(dataset_dir='/store/tellus/train',
                  output='/store/tellus/train.pqt'):
    rows = pipe(
        concatv(
            get_train_row(
                base_path=dataset_dir,
                label_dir="positive",
                label=1,
            ),
            get_train_row(
                base_path=dataset_dir,
                label_dir="negative",
                label=0,
            ),
        ),
        list
    )
    df = pd.DataFrame(rows)
    df = df.sort_values(by=['id'])
    df = df.reset_index()
    df.to_parquet(output, compression='gzip')
    return output


@skip_if_exists("output")
def load_test_df(dataset_dir='/store/tellus/test',
                 output='/store/tellus/test.pqt'):
    rows = get_test_row(
        base_path=dataset_dir,
    )
    df = pd.DataFrame(rows)
    df = df.sort_values(by=['id'])
    df = df.reset_index()
    df.to_parquet(output, compression='gzip')
    return output


def image_to_tensor(path):
    image = io.imread(
        path,
    )
    if len(image.shape) == 2:
        h, w = image.shape
        image = img_as_float(image.reshape(h, w, -1)).astype(np.float32)
    tensor = ToTensor()(image)
    return tensor


class TellusDataset(Dataset):
    def __init__(self, df, has_y=True):
        self.has_y = has_y
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id = row['id']
        palsar_after = image_to_tensor(
            row['palsar_after'],
        )
        palsar_before = image_to_tensor(
            row['palsar_before'],
        )

        palsar = torch.cat(
            [palsar_before, palsar_after],
            dim=0,
        )

        if self.has_y:
            aug = Augment()

            landsat_after = image_to_tensor(
                row['landsat_after']
            )
            landsat_before = image_to_tensor(
                row['landsat_before']
            )
            landsat = torch.cat(
                [landsat_before, landsat_after],
                dim=0,
            )

            return {
                'id': id,
                'palsar': palsar,
                'landsat': landsat,
                'label': row['label'],
            }
        else:
            return {
                'id': id,
                'palsar': palsar,
            }


class ChunkSampler(Sampler):
    def __init__(self,  epoch_size, len_indices, shuffle=True, start_at=0):
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
        self.chunk_idx = start_at

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


def kfold(df, n_splits, random_state=0):
    kf = KFold(n_splits, random_state=random_state, shuffle=True)
    pos_df = df[df['label'] == 1]
    neg_df = df[df['label'] == 0]
    dataset = TellusDataset(df, has_y=True)

    splieted = pipe(
        zip(kf.split(pos_df), kf.split(neg_df)),
        map(lambda x: {
            "train_pos": pos_df.index[x[0][0]],
            "val_pos": pos_df.index[x[0][1]],
            "train_neg": neg_df.index[x[1][0]],
            "val_neg": neg_df.index[x[1][1]],
        }),
        map(valmap(lambda x: Subset(dataset, x))),
        list
    )

    return splieted


class Augment(object):
    def __init__(self):
        augs = [
        ]
        self.augs = pipe(
            augs,
            map(lambda a: random.choice([lambda x: x, a])),
            list,
        )
        self.transform = Compose([
            ToPILImage(),
            *self.augs,
            ToTensor(),
        ])

    def __call__(self, t):
        return self.transform(t)


@curry
def batch_aug(aug, batch, ch=3):
    return pipe(
        batch,
        map(lambda x: [aug(x[0:ch, :, :]), aug(x[ch:2*ch, :, :])]),
        map(lambda x: torch.cat(x, dim=0)),
        list,
        torch.stack
    )



@curry
def batch_crop(position, batch):
    _, _, h, w = batch.size()
    crop_h = (h * 3) // 4
    crop_w = (w * 3) // 4
    start_h = position[0] * w//4
    end_h = start_h + crop_h
    start_w = position[1] * w//4
    end_w = start_w + crop_w

    return batch[:, :, start_h:end_h, start_w:end_w]
