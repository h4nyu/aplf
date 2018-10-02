from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk, curry, merge, first
import pandas as pd
from aplf import config
import torch
from torch.utils.data import DataLoader
from aplf.kaggle.tgs.preprocess import rl_enc, cleanup, rle_decode, add_mask_size, add_noise, add_cover, divide_by_cover
from aplf.kaggle.tgs.dataset import TgsSaltDataset, load_dataset_df
from tensorboardX import SummaryWriter
import torchvision.utils as vutils


def test_dataset():
    dataset_df = load_dataset_df('/store/kaggle/tgs')
    dataset = TgsSaltDataset(dataset_df)
    assert len(dataset) == 4000


def test_cleanup():
    dataset_df = load_dataset_df('/store/kaggle/tgs')
    dataset_df = cleanup(dataset_df)
    assert len(dataset_df) == 3808


def test_rle_decode():
    dataset_df = load_dataset_df('/store/kaggle/tgs')
    idx = dataset_df.index.get_loc('1fba03699e')
    sample = dataset_df.iloc[idx]
    output = rle_decode(sample['rle_mask'], (101, 101))
    output = torch.FloatTensor(output).view(1, 101, 101)
    assert output.sum() == 487


def test_add_mask_size():
    dataset_df = load_dataset_df('/store/kaggle/tgs')
    dataset_df = add_mask_size(dataset_df)
    print(dataset_df)

    idx = dataset_df.index.get_loc('1fba03699e')
    sample = dataset_df.iloc[idx]
    output = rle_decode(sample['rle_mask'], (101, 101))
    output = torch.FloatTensor(output).view(1, 101, 101)
    assert output.sum() == 487



def test_add_noise():
    dataset_df = load_dataset_df(
        '/store/kaggle/tgs',
        'train.csv',
    )
    dataset = TgsSaltDataset(dataset_df)

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True
    )
    sample = pipe(
        dataloader,
        first
    )['image']
    noised = add_noise(sample)

    writer = SummaryWriter(f'{config["TENSORBORAD_LOG_DIR"]}/test')
    dataset_df = load_dataset_df('/store/kaggle/tgs')
    writer.add_image(
        f"add_noise",
        vutils.make_grid([*sample, *noised]),
    )


def test_divide_by_cover():
    dataset_df = load_dataset_df('/store/kaggle/tgs')
    dataset_df = add_cover(dataset_df)
    assert dataset_df.iloc[0]['cover'] == 0

    dfs = divide_by_cover(dataset_df, 4)
    assert len(dfs) == 4


