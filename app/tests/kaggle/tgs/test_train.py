from aplf.kaggle.tgs.train import train
import numpy as np
from torch.utils.data import Subset
from datetime import datetime
from aplf.kaggle.tgs.dataset import TgsSaltDataset, load_dataset_df
from sklearn.model_selection import train_test_split
from aplf import config
import uuid


def test_train():
    train_config = {
        'epochs': 1,
        'labeled_batch_size': 32,
        'no_labeled_batch_size': 16,
        'model_type': 'HUNet',
        'model_kwargs':{
            'feature_size': 16,
            'depth': 3,
        },
        'ema_decay': 0.1,
        'consistency': 1,
        'consistency_rampup': 10,
        'cyclic_period': 5,
        'milestones': [(0, 1)],
        'erase_num': 3,
    }
    dataset_df = load_dataset_df(
        '/store/kaggle/tgs',
        'train.csv'
    )
    train_df, val_df = train_test_split(dataset_df, test_size=0.2)

    unsupervised_dataset = load_dataset_df(
        '/store/kaggle/tgs',
        'sample_submission.csv'
    )

    dataset=TgsSaltDataset(
        train_df,
        has_y=True
    )
    train_set = Subset(
        dataset=dataset,
        indices=np.arange(100)
    )

    val_set = Subset(
        dataset=dataset,
        indices=np.arange(100)
    )

    train(
        **train_config,
        model_path='/store/tmp/mock_model.pt',
        train_set=train_set,
        val_set=val_set,
        no_labeled_set=train_set,
        log_dir=f'{config["TENSORBORAD_LOG_DIR"]}/test',
    )
