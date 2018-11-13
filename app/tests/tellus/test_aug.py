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


@pytest.mark.parametrize("alpha", [0.1, 0.5, 1])
@pytest.mark.parametrize("sigma", [1, 10, 50])
def test_elastic(dataset, alpha, sigma):
    aug = Augment(
        transform=A.ElasticTransform(
            p=1,
            alpha=alpha,
            sigma=sigma,
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
        f"ElasticTransform/alpha/{alpha}/sigma/{sigma}/palsar",
        vutils.make_grid(
            [
                *aug_sample['palsar_before'],
                *no_aug_sample['palsar_before'],
                *aug_sample['palsar_after'],
                *no_aug_sample['palsar_after'],
            ]
        ),
    )


@pytest.mark.parametrize("var_limit", [(1, 10), (10, 50), (0, 255)])
def test_gauss_noise(dataset, var_limit):
    aug = Augment(
        transform=A.Compose([
            A.FromFloat(),
            A.GaussNoise(
                p=1,
                var_limit=var_limit
            ),
            A.ToFloat(),
        ], p=1)
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
        f"GaussNoise/var_limit/{var_limit}/palsar",
        vutils.make_grid(
            [
                *aug_sample['palsar_before'],
                *no_aug_sample['palsar_before'],
                *aug_sample['palsar_after'],
                *no_aug_sample['palsar_after'],
            ]
        ),
    )


@pytest.mark.parametrize("blur_limit", [3, 4, 7])
def test_median_blur(dataset, blur_limit):
    aug = Augment(
        transform=A.MedianBlur(
            p=1,
            blur_limit=blur_limit
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
        f"MedianBlur/blur_limit/{blur_limit}/palsar",
        vutils.make_grid(
            [
                *aug_sample['palsar_before'],
                *no_aug_sample['palsar_before'],
                *aug_sample['palsar_after'],
                *no_aug_sample['palsar_after'],
            ]
        ),
    )


@pytest.mark.parametrize("distort_limit", [0.05, 0.03, 0.01])
@pytest.mark.parametrize("shift_limit", [0.05, 0.03])
def test_optical_distorsion(dataset, distort_limit, shift_limit):
    aug = Augment(
        transform=A.OpticalDistortion(
            p=1,
            distort_limit=distort_limit,
            shift_limit=shift_limit
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
        f"OpticalDistortion/distort_limit/{distort_limit}/shift_limit/{shift_limit}/palsar",
        vutils.make_grid(
            [
                *aug_sample['palsar_before'],
                *no_aug_sample['palsar_before'],
                *aug_sample['palsar_after'],
                *no_aug_sample['palsar_after'],
            ]
        ),
    )
