from aplf import config
from aplf.tellus.predict import predict
import pandas as pd
from aplf.tellus.model import Net
from aplf.tellus.data import load_test_df, TellusDataset
import pytest
import torch


def test_predict():
    df_path = load_test_df(
        dataset_dir='/store/tellus/test',
        output='/store/tmp/test.pqt'
    )
    df = pd.read_parquet(df_path)[:100]
    dataset = TellusDataset(
        df,
        has_y=False
    )
    model = Net(
        feature_size=64,
        resize=120,
        pad=4,
    )
    model_paths = ['/store/tmp/model.pt']
    torch.save(model, model_paths[0])
    predicted_df = predict(
        model_paths=model_paths,
        log_dir=f'{config["TENSORBORAD_LOG_DIR"]}/test',
        dataset=dataset,
        out_path="/store/tmp/submission.tsv",
        log_interval=1
    )
    #  print(predicted_df)
    #  assert len(predicted_df) == len(dataset_df)
