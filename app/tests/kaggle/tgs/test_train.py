from aplf.kaggle.tgs.train import train
from datetime import datetime
from aplf.kaggle.tgs.dataset import TgsSaltDataset, load_dataset_df
from sklearn.model_selection import train_test_split
from aplf import config
import uuid


def test_train():
    dataset_df = load_dataset_df(
        '/store/kaggle/tgs',
        'train.csv'
    )
    train_df, val_df = train_test_split(dataset_df, test_size=0.2)

    unsupervised_dataset = load_dataset_df(
        '/store/kaggle/tgs',
        'sample_submission.csv'
    )
    train(
        model_path='/store/tmp/model.pt',
        train_dataset=TgsSaltDataset(
            train_df,
            has_y=True
        ),
        val_dataset=TgsSaltDataset(
            val_df,
            has_y=True
        ),
        no_labeled_dataset=TgsSaltDataset(
            unsupervised_dataset,
            has_y=False
        ),
        epochs=2,
        feature_size=16,
        labeled_batch_size=12,
        no_labeled_batch_size=16,
        patience=5,
        base_size=5,
        ema_decay=0.9,
        consistency=0.5,
        consistency_rampup=10,
        depth=3,
        log_dir=f'{config["TENSORBORAD_LOG_DIR"]}/test',
    )
