from aplf.kaggle.tgs.train import base_train
import numpy as np
from torch.utils.data import Subset
from datetime import datetime
from aplf.kaggle.tgs.dataset import TgsSaltDataset, load_dataset_df
from sklearn.model_selection import train_test_split
from aplf import config
import uuid


def test_train():
    train_config = {
        'epochs': 2,
        'batch_size': 16,
        'no_label_batch_size': 4,
        'model_type': 'SHUNet',
        'size_diff': (-10, 10),
        "resize":(118, 118),
        'model_kwargs': {
            'feature_size': 64,
        },
        'consistency_loss_wight': 10,
        'center_loss_weight': 0.3,
        'seg_loss_weight': 0.5,
    }
    dataset_df = load_dataset_df(
        '/store/kaggle/tgs',
        'train.csv'
    )
    train_df, val_df = train_test_split(dataset_df, test_size=1/8)

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

    base_train(
        **train_config,
        model_path='/store/tmp/mock_model.pt',
        train_set=train_set,
        val_set=val_set,
        seg_set=train_set,
        no_lable_set=train_set,
        log_dir=f'{config["TENSORBORAD_LOG_DIR"]}/test',
    )
