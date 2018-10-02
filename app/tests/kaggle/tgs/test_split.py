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


def test_split():
    dataset_df = load_dataset_df('/store/kaggle/tgs')
    assert len(dataset_df) == 4000
    dataset = TgsSaltDataset(dataset_df)
    assert len(dataset) == 4000
    assert len(dataset[0]) == 4


