from aplf.kaggle.tgs.predict import predict
from aplf.kaggle.tgs.dataset import TgsSaltDataset, load_dataset_df
from aplf.kaggle.tgs.model import UNet, HUNet
from aplf import config
from sklearn.model_selection import train_test_split
import torch
import pytest


@pytest.mark.parametrize("csv_fn, has_y", [
    ('sample_submission.csv', False),
    ('train.csv', True),
])
def test_predict(csv_fn, has_y):
    dataset_df = load_dataset_df(
        '/store/kaggle/tgs',
        csv_fn
    ).sample(10)
    dataset = TgsSaltDataset(dataset_df, has_y=has_y)
    model = HUNet()
    model_paths = ['/store/tmp/model.pt']
    torch.save(model, model_paths[0])
    predicted_df = predict(
        model_paths=model_paths,
        log_dir=f'{config["TENSORBORAD_LOG_DIR"]}/test',
        hdf5_path="/store/tmp/out.hdf5",
        dataset=dataset,
        log_interval=1
    )
    assert len(predicted_df) == len(dataset_df)
