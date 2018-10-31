from pathlib import Path
from aplf.tellus.data import load_train_df, get_train_row, TellusDataset, kfold, ChunkSampler, get_test_row, load_test_df
from torch.utils.data import DataLoader
import pandas as pd
import pytest

from cytoolz.curried import keymap, filter, pipe, merge, map, compose, concatv, first, take, concat
from torch.utils.data import Subset
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from aplf import config


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
    palsar = sample['palsar']


    writer = SummaryWriter(f'{config["TENSORBORAD_LOG_DIR"]}/test/aug')
    writer.add_image(
        f"hflip/original",
        vutils.make_grid([*palsar[:, 0:1, :, :], *palsar[:, 1:2, :, :]]),
    )

    writer.add_image(
        f"hflip/aug",
        vutils.make_grid([*palsar[:, 0:1, :, :], *palsar[:, 1:2, :, :]]),
    )
