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
        self.y = df['Score'].values
        self.x = df.drop('Score', axis=1).values


    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> t.Tuple[Tensor, Tensor]:
        x = self.x[idx]
        y = self.y[idx]
        return tensor(x, dtype=float32), tensor(y, dtype=float32)


def create_dataset(df) -> lgbm.Dataset:
    y = df['Score'].values
    x = df.drop('Score', axis=1).values
    return lgbm.Dataset(data=x, label=y)

def save_model(model, path:str):
    torch.save(model.state_dict(), path)
