from aplf import config
import torch
from aplf.kaggle.tgs.preprocess import rl_enc, cleanup, rle_decode, add_mask_size
from aplf.kaggle.tgs.dataset import TgsSaltDataset, load_dataset_df
from aplf.kaggle.tgs.metric import iou
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np


def test_dataset():
    dataset_df = load_dataset_df('/store/kaggle/tgs')
    dataset = TgsSaltDataset(dataset_df)
    scores = []
    for i in range(12):
        sample = dataset[i]
        score = iou(sample['mask'].numpy(), sample['mask'].numpy())
        scores.append(score)
    assert np.round(np.mean(scores), decimals=3) == 1.0
