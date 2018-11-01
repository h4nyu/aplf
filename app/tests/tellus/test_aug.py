from pathlib import Path
from aplf.tellus.data import load_train_df, get_train_row, TellusDataset, kfold, ChunkSampler, get_test_row, load_test_df, Augment
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
    batch_sample = pipe(
        loader,
        first,
    )
    aug = Augment()
    writer = SummaryWriter(f'{config["TENSORBORAD_LOG_DIR"]}/test/aug')
    for i, s in enumerate( batch_sample['palsar']):
        writer.add_image(
            f"hflip/aug",
            vutils.make_grid(
                [aug(s[0:1, :, :]), s[0:1, :, :], aug(s[1:2, :, :]), s[1:2, :, :]]
            ),
            i
        )
