from aplf.kaggle.tgs.dataset import TgsSaltDataset, load_dataset_df
from cytoolz.curried import keymap, filter, pipe, merge, map, concat
from torchvision.transforms import (
    RandomRotation,
    ToPILImage,
    Compose,
    ToTensor,
    CenterCrop,
    RandomAffine,
    TenCrop,
    RandomApply,
    RandomHorizontalFlip,
    RandomVerticalFlip

)
from tensorboardX import SummaryWriter
import numpy as np
import torchvision.utils as vutils

from aplf import config


def test_dataset():
    dataset_df = load_dataset_df('/store/kaggle/tgs')
    assert len(dataset_df) == 4000
    dataset = TgsSaltDataset(dataset_df)
    assert len(dataset) == 4000
    assert len(dataset[0]) == 4


def test_flip():
    writer = SummaryWriter(f'{config["TENSORBORAD_LOG_DIR"]}/test')
    dataset_df = load_dataset_df('/store/kaggle/tgs')
    dataset = TgsSaltDataset(dataset_df)
    writer.add_image(
        f"flip",
        vutils.make_grid(
            pipe(range(8),
                 map(lambda x: dataset[12]),
                 map(lambda x: [x['image'], x['mask']]),
                 concat,
                 list)
        ),
    )
