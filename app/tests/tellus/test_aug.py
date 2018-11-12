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
import torchvision.utils as vutils
from aplf import config
from torchvision import datasets, models, transforms
import albumentations.augmentations.functional as AF
from torchvision.transforms.functional import (
    adjust_brightness
)

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


@pytest.mark.parametrize("num_steps", [1, 5])
@pytest.mark.parametrize("distort_limit", [0.1])
def test_grid_distorsion(dataset, num_steps, distort_limit):
    transform = ToTensor()
    loader = DataLoader(
        dataset,
        batch_size=1,
        transform=transform,
    )

    sample = first(loader)
    print(sample['palsar_before'].size())

    #  aug = Augment(
    #      [
    #          lambda x: A.GridDistortion(
    #              num_steps=num_steps,
    #              distort_limit=distort_limit,
    #              p=1,
    #          )(image=x)['image'].reshape(1, 40, 40)
    #      ]
    #  )
    #
    #  writer = SummaryWriter(f'{config["TENSORBORAD_LOG_DIR"]}/test/aug')
    #  writer.add_image(
    #      f"GridDistortion/palsar",
    #      vutils.make_grid(
    #          [
    #              *batch_aug(aug, sample['palsar'], ch=1)[:, 0:1, :, :],
    #              *batch_aug(aug, sample['palsar'], ch=1)[:, 1:2, :, :],
    #              *sample['palsar'][:, 0:1, :, :],
    #              *sample['palsar'][:, 1:2, :, :],
    #          ]
    #      ),
    #  )
