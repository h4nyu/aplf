from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk, curry, merge, first
import torch.nn.functional as F
from aplf import config
import torch
from torch.utils.data import DataLoader
from aplf.kaggle.tgs.preprocess import rl_enc, cleanup, rle_decode, add_mask_size, add_noise, RandomErasing
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

def test_erase():
    dataset_df = load_dataset_df(
        '/store/kaggle/tgs',
        'train.csv',
    )
    dataset = TgsSaltDataset(
        dataset_df,
        has_y=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True
    )
    sample = pipe(
        dataloader,
        first
    )['image']
    random_erase = RandomErasing()

    noised = add_noise(
        sample,
        erase_num=5,
        erase_p=1
    )

    writer = SummaryWriter(f'{config["TENSORBORAD_LOG_DIR"]}/test')
    dataset_df = load_dataset_df('/store/kaggle/tgs')
    writer.add_image(
        f"random_erase",
        vutils.make_grid([*sample, *noised]),
    )


def test_pool():
    dataset_df = load_dataset_df(
        '/store/kaggle/tgs',
        'train.csv',
    )
    dataset = TgsSaltDataset(
        dataset_df,
        has_y=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True
    )
    sample = pipe(
        dataloader,
        first
    )['mask']
    interpolated = F.max_pool2d(sample, kernel_size=101)

    writer = SummaryWriter(f'{config["TENSORBORAD_LOG_DIR"]}/test')
    dataset_df = load_dataset_df('/store/kaggle/tgs')
    writer.add_image(
        f"mask",
        vutils.make_grid([*sample]),
        0 
    )

    writer.add_image(
        f"pooled",
        vutils.make_grid([*interpolated]),
        0
    )
