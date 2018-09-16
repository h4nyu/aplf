from aplf.kaggle.tgs.train import train
from datetime import datetime
from aplf.kaggle.tgs.dataset import TgsSaltDataset, load_dataset_df
from sklearn.model_selection import train_test_split
from aplf import config
import uuid


def test_train():
    dataset_df = load_dataset_df('/store/kaggle/tgs')
    train_df, val_df = train_test_split(dataset_df)
    output_dir = '/store/tmp'
    train(
        model_path=f"{output_dir}/model.pt",
        train_dataset=TgsSaltDataset(train_df),
        val_dataset=TgsSaltDataset(val_df),
        epochs=1000,
        batch_size=32,
        feature_size=32,
        patience=5,
        base_size=5,
        log_dir=f'{config["TENSORBORAD_LOG_DIR"]}/test',
    )
