from logging import getLogger
from pathlib import Path
from torch.utils.data import Dataset
import typing as t
import pandas as pd
import numpy as np
from torch import Tensor, tensor, float32
from typing_extensions import Protocol
from sklearn.model_selection import KFold
from pathlib import Path
import torch
import lightgbm as lgbm
from aplf.utils.decorators import skip_if
from pathlib import Path
from .models import Model


logger = getLogger("takeda.data")


@skip_if(
    lambda *args: Path(args[1]).is_file(),
    lambda *args: args[1],
)
def csv_to_pkl(
    in_path: str,
    out_path: str,
) -> t.Any:
    df = pd.read_csv(in_path)
    df = df.set_index('ID')
    df.to_pickle(out_path)
    return out_path

@skip_if(
    lambda *args: Path(args[1]).is_file(),
    lambda *args: args[1],
)
def extracet_summary(
    in_path: str,
    out_path: str,
) -> t.Any:
    df = pd.read_pickle(in_path)
    ds = df.describe()
    ds.to_json(out_path)
    return out_path

@skip_if(
    lambda *args: Path(args[1]).is_file(),
    lambda *args: args[1],
)
def extract_col_type(
    in_path: str,
    out_path: str,
) -> t.Any:
    df = pd.read_json(in_path)
    for i in df.iterrows():
        print(i)
    return out_path


@skip_if(
    lambda *args: Path(args[2]).is_file(),
    lambda *args: args[2],
)
def compare_feature(
    source_path: str,
    dest_path: str,
    out_path: str,
) -> t.Any:
    source = pd.read_json(source_path)
    dest = pd.read_json(dest_path)
    interaction = set(dest.columns) & set(source.columns)
    df = dest[interaction] - source[interaction]
    df.to_json(out_path)
    return out_path

@skip_if(
    lambda *args: Path(args[2]).is_file(),
    lambda *args: args[2],
)
def detect_col_type(
    in_path: str,
    out_path: str,
) -> t.Any:
    souce_df = pd.read_json(source_path)
    return out_path

@skip_if(
    lambda *args: Path(args[2]).is_file(),
    lambda *args: args[2],
)
def extract_score_distorsion(
    in_path: str,
    out_path: str,
) -> t.Any:
    souce_df = pd.read_pickle(source_path)
    score = souce_df['Score']
    return out_path

def kfold(
    dataset: Dataset,
    n_splits: int,
    random_state: int = 0
) -> t.List[t.Tuple[t.Sequence[int], t.Sequence[int]]]:
    kf = KFold(n_splits, random_state=random_state, shuffle=True)
    return list(kf.split(dataset))


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

    def __getitem__(self, idx: int) -> Tensor:
        x = self.x[idx]
        return tensor(x, dtype=float32)


def create_dataset(df: t.Any) -> t.Tuple[t.Any, t.Any]:
    y = df['Score'].values
    x = df.drop('Score', axis=1).values
    return lgbm.Dataset(x, y)


def save_model(model: Model, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_model(path: str) -> Model:
    model = Model(
        size_in=3805,
    )
    model.load_state_dict(torch.load(path))
    return model


def save_submit(
    df,
    preds,
    path,
) -> Path:
    submit_df = pd.DataFrame({
        'Score': preds
    }, index=df.index)
    submit_df.to_csv(path, header=False)
    return submit_df
