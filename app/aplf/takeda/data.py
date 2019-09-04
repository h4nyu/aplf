from logging import getLogger
from pathlib import Path
from torch.utils.data import Dataset
import typing as t
import pandas as pd
import numpy as np
from torch import Tensor, tensor, float32
from typing_extensions import Protocol
from sklearn.model_selection import KFold
import torch
import lightgbm as lgbm
from .models import Model


logger = getLogger("takeda.data")


def read_csv(path: str) -> t.Any:
    df = pd.read_csv(path)
    df = df.set_index('ID')
    return df


def kfold(
    dataset: Dataset,
    n_splits: int,
    random_state: int = 0
) -> t.List[t.Tuple[t.Sequence[int], t.Sequence[int]]]:
    kf = KFold(n_splits, random_state=random_state, shuffle=True)
    return list(kf.split(dataset))


def add_noise(df, unique_threshold=2):
    indices = []
    for idx in range(len(df.columns)):
        print(df.iloc[:, idx].unique())
        print(idx)
        if len(df.iloc[:, idx].unique()) > unique_threshold:
            indices.append(idx)


class TakedaDataset(Dataset):
    def __init__(self, df: t.Any) -> None:
        self.x = df.drop('Score', axis=1).values
        self.y = df['Score'].values


    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> t.Tuple[Tensor, Tensor]:
        x = self.x[idx]
        y = self.y[idx]
        return tensor(x, dtype=float32), tensor(y, dtype=float32)

class TakedaPredDataset(Dataset):
    def __init__(self, df: t.Any) -> None:
        self.x = df.values

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> t.Tuple[Tensor, Tensor]:
        x = self.x[idx]
        return tensor(x, dtype=float32)


def create_dataset(df) -> lgbm.Dataset:
    y = df['Score'].values
    x = df.drop('Score', axis=1).values
    return lgbm.Dataset(data=x, label=y)

def save_model(model, path:str):
    torch.save(model.state_dict(), path)


def load_model(path:str):
    model = Model(
        size_in=3805,
    )
    model.load_state_dict(torch.load(path))
    return model

def save_submit(df, preds, path):
    submit_df = pd.DataFrame({
        'Score': preds
    }, index=df.index)
    submit_df.to_csv(path, header=False)
    return submit_df
