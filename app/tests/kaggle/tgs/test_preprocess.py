from aplf import config
import torch
from aplf.kaggle.tgs.preprocess import rl_enc, cleanup, rle_decode, add_mask_size
from aplf.kaggle.tgs.dataset import TgsSaltDataset, load_dataset_df
from tensorboardX import SummaryWriter


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
