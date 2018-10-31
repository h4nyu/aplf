from aplf import config
from aplf.tellus.predict import predict
import pandas as pd
from aplf.tellus.data import load_test_df, TellusDataset
import pytest
import torch


def test_predict():
    df_path = load_test_df(
        dataset_dir='/store/tellus/test',
        output='/store/tmp/test.pqt'
    )
    df = pd.read_parquet(df_path)
    dataset = TellusDataset(
        df,
        has_y=False
    )
    model_dirs = ['/store/tellus/output/0082']
    predicted_df = predict(
        model_dirs=model_dirs,
        dataset=dataset,
        out_path="/store/tmp/submission-mean.tsv",
    )
    assert len(predicted_df) == len(df)
