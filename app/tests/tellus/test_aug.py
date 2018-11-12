from pathlib import Path
from aplf.tellus.data import load_train_df, get_train_row, TellusDataset, kfold, ChunkSampler, get_test_row, load_test_df
from torch.utils.data import DataLoader
import numpy as np
import cv2
import pandas as pd
import pytest
from urllib.request import urlopen
from cytoolz.curried import keymap, filter, pipe, merge, map, compose, concatv, first, take, concat
from torch.utils.data import Subset
from tensorboardX import SummaryWriter
import albumentations as A
from albumentations.torch import ToTensor
import torchvision.utils as vutils
from aplf import config
from torchvision import datasets, models, transforms
import albumentations.augmentations.functional as AF
from torchvision.transforms.functional import (
    adjust_brightness
)
from aplf.tellus.aug import Augment
import albumentations as A

from torchvision.transforms import ToTensor


@pytest.fixture
def dataset():
    output = load_train_df(
        dataset_dir='/store/tellus/train',
        output='/store/tmp/train.pqt'
    )
    df = pd.read_parquet(output)
    return TellusDataset(
        df=df,
        has_y=True,
    )


@pytest.mark.parametrize("num_steps", [1, 2,  5, 10])
@pytest.mark.parametrize("distort_limit", [0.01, 0.1, 0.3])
def test_grid_distorsion(dataset, num_steps, distort_limit):
    aug = Augment(
        transform=A.GridDistortion(
            p=1,
            num_steps=num_steps,
            distort_limit=distort_limit,
        )
    )
    dataset.transform = aug
    loader = DataLoader(
        dataset,
        batch_size=1,
    )
    aug_sample = first(loader)

    dataset.transform = ToTensor()
    loader = DataLoader(
        dataset,
        batch_size=1,
    )
    no_aug_sample = first(loader)

    writer = SummaryWriter(f'{config["TENSORBORAD_LOG_DIR"]}/test/aug')
    writer.add_image(
        f"GridDistortion/num_steps/{num_steps}/distort_limit/{distort_limit}/palsar",
        vutils.make_grid(
            [
                *aug_sample['palsar_before'],
                *no_aug_sample['palsar_before'],
                *aug_sample['palsar_after'],
                *no_aug_sample['palsar_after'],
            ]
        ),
    )
