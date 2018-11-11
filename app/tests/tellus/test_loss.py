import torch.nn as nn
from cytoolz.curried import keymap, filter, pipe, merge, map, compose, concatv, first, take, concat
from tensorboardX import SummaryWriter
from aplf import config
from aplf.tellus.predict import predict
from aplf.tellus.data import load_train_df, get_train_row, TellusDataset, kfold, ChunkSampler, get_test_row, load_test_df, Augment, batch_aug
from torch.utils.data import DataLoader
from aplf.losses import SSIM
import pandas as pd
from aplf.tellus.data import load_test_df, TellusDataset
import pytest
import torch


def test_ssim():

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
    cri = nn.MSELoss(size_average=True)
    same = cri(sample['landsat'], (sample['landsat']))
    diff = cri(sample['landsat'], (sample['landsat'] / 2))
    assert same < diff
    cri = SSIM(window_size=3)
    same = - cri(sample['landsat'], (sample['landsat']))
    diff = - cri(sample['landsat'], (sample['landsat'] / 2))
    assert same < diff
