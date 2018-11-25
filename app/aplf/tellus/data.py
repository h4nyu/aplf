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

        self.transforms = [
            lambda x:x,
            hflip,
            vflip,
            lambda x: rotate(x, 90),
            lambda x: rotate(x, -90),
            lambda x: rotate(x, -180),
        ]

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
            #  hflip,
            #  vflip,
            #  lambda x: rotate(x, 90),
            #  lambda x: adjust_brightness(x, 2),
            #  lambda x: adjust_contrast(x, 2),

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


class RandomErasing(object):

    def __init__(self, p=0.5,  sl=0.01, sh=0.05, r1=1, num=1, mean=[0, 0.0, 0.0]):
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.num = num
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img
        for attempt in range(self.num):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]

        return img


@curry
def batch_aug_concat(aug, batch, ch=3):
    return pipe(
        batch,
        map(lambda x: [aug(x[0:ch, :, :]), aug(x[ch:2*ch, :, :])]),
        map(lambda x: torch.cat(x, dim=0)),
        list,
        torch.stack
    )


@curry
def batch_aug(aug, batch):
    return pipe(
        batch,
        map(aug),
        list,
        torch.stack
    )


def get_spatical_shuffle(batch_size, grid=(2, 2)):
    idx_arr = np.arange(batch_size)
    return []


def batch_spatical_shuffle(batch, indices):
    b, c, w, h = batch.size()
    block00 = batch[indices[0], :, 0:w//2, 0:h//2]
    block01 = batch[indices[1], :, 0:w//2, h//2:h]
    block11 = batch[indices[2], :, w//2:w, h//2:h]
    block10 = batch[indices[3], :, w//2:w, 0:h//2]

    auged = torch.zeros(batch.size())
    auged[:, :, 0:w//2, 0:h//2] = block00
    auged[:, :, w//2:w, 0:h//2] = block10
    auged[:, :, w//2:w, h//2:h] = block11
    auged[:, :, 0:w//2, h//2:h] = block01
    return auged
