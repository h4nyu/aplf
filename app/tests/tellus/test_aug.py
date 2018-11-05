from pathlib import Path
from aplf.tellus.data import load_train_df, get_train_row, TellusDataset, kfold, ChunkSampler, get_test_row, load_test_df, Augment, batch_aug
from torch.utils.data import DataLoader
import numpy as np
import cv2
import pandas as pd
import pytest
from urllib.request import urlopen
from cytoolz.curried import keymap, filter, pipe, merge, map, compose, concatv, first, take, concat
from torch.utils.data import Subset
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from aplf import config
from torchvision import datasets, models, transforms
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
    adjust_brightness
)


def test_test_dataset():
    output = load_train_df(
        dataset_dir='/store/tellus/train',
        output='/store/tmp/train.pqt'
    )
    df = pd.read_parquet(output)
    dataset = TellusDataset(
        df=df,
        has_y=True,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=False,
    )
    sample = pipe(
        loader,
        first,
    )
    aug = Augment()

    writer = SummaryWriter(f'{config["TENSORBORAD_LOG_DIR"]}/test/aug')
    writer.add_image(
        f"brightness/landsat",
        vutils.make_grid(
            [
                *batch_aug(aug, sample['landsat'], ch=3)[:, 0:3, :, :],
                *batch_aug(aug, sample['landsat'], ch=3)[:, 3:6, :, :],
                *sample['landsat'][:, 0:3, :, :],
                *sample['landsat'][:, 3:6, :, :],
            ]
        ),
    )

    writer.add_image(
        f"brightness/palsar",
        vutils.make_grid(
            [
                *batch_aug(aug, sample['palsar'], ch=1)[:, 0:1, :, :],
                *batch_aug(aug, sample['palsar'], ch=1)[:, 1:2, :, :],
                *sample['palsar'][:, 0:1, :, :],
                *sample['palsar'][:, 1:2, :, :],
            ]
        ),
    )
