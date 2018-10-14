from aplf.kaggle.tgs.dataset import TgsSaltDataset, load_dataset_df
from cytoolz.curried import keymap, filter, pipe, merge, map, concat
from torch.utils.data import Subset
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
from aplf.kaggle.tgs.dataset import get_segment_indices


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

def test_segment_set():
    dataset_df = load_dataset_df('/store/kaggle/tgs')
    dataset = TgsSaltDataset(dataset_df)
    segment_indices = get_segment_indices(dataset, range(10, 15))
    assert segment_indices == [10, 12, 14]
    print(segment_indices)
