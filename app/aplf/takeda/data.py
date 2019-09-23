import pickle
from logging import getLogger
from pathlib import Path
from torch.utils.data import Dataset
import typing as t
import pandas as pd
import numpy as np
from numpy.random import random_integers, randint
from torch import Tensor, tensor, float32, save, float64
from typing_extensions import Protocol
from sklearn.model_selection import KFold
from pathlib import Path
import torch
import lightgbm as lgbm
from aplf.utils.decorators import skip_if
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from .models import Model



logger = getLogger("takeda.data")


@skip_if(
    lambda *args: Path(args[1]).is_file(),
    lambda *args: pd.read_pickle(args[1]),
)
def csv_to_pkl(
    in_path: str,
    out_path: str,
) -> t.Any:
    df = pd.read_csv(in_path)
    df = df.set_index('ID')
    df.to_pickle(out_path)
    return df

@skip_if(
    lambda *args: Path(args[1]).is_file(),
    lambda *args: pd.read_json(args[1]),
)
def extracet_summary(
    df: t.Any,
    out_path: str,
) -> t.Any:
    ds = df.describe()
    ds.to_json(out_path)
    return df

@skip_if(
    lambda *args: Path(args[1]).is_file(),
    lambda *args: pd.read_json(args[1]),
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
    souce_df = pd.read_json(in_path)
    return out_path

@skip_if(
    lambda *args: Path(args[2]).is_file(),
    lambda *args: args[2],
)
def extract_score_distorsion(
    in_path: str,
    out_path: str,
) -> t.Any:
    souce_df = pd.read_pickle(in_path)
    score = souce_df['Score']
    return out_path

@skip_if(
    lambda *args: Path(args[2]).is_file(),
    lambda *args: args[2],
)
def create_dateaset(
    in_path: str,
    out_path: str,
) -> t.Any:
    df = pd.read_pickle(in_path)
    dataset = TakedaDataset(df)
    with open(out_path, 'wb') as f:
        pickle.dump(dataset, f)
    score = df['Score']
    return out_path

def kfold(
    dataset: Dataset,
    n_splits: int,
    random_state: int = 0
) -> t.List[t.Tuple[t.Sequence[int], t.Sequence[int]]]:
    kf = KFold(n_splits, random_state=random_state, shuffle=True)
    return list(kf.split(dataset))


class TakedaDataset(Dataset):
    def __init__(self, df: t.Any, ignore_columns:t.Set[str]=set()) -> None:
        x_columns = sorted(set(df.columns) - ignore_columns - {'Score'})
        self.x_columns = x_columns
        self.x = df[x_columns].values
        self.y = df['Score'].values
        self.df = df

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> t.Tuple[Tensor, Tensor]:
        x = self.x[idx]
        y = self.y[idx]
        return tensor(x, dtype=float64), tensor(y, dtype=float64)


class TakedaPredDataset(Dataset):
    def __init__(self, df: t.Any) -> None:
        self.x = df.values

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> t.Tuple[Tensor, Tensor]:
        x = self.x[idx]
        return tensor(x, dtype=float64), tensor(0, dtype=float64)


def create_dataset(df: t.Any) -> t.Tuple[t.Any, t.Any]:
    y = df['Score'].values
    x = df.drop('Score', axis=1).values
    return lgbm.Dataset(x, y)


def save_model(model: Model, path: str) -> None:
    save(model, path)


def load_model(path: str) -> Model:
    return torch.load(path)

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


def dump_hist_plot(series:t.Any, path:str) -> None:
    if Path(path).is_file():
        return
    ax = series.hist(bins=100)  # s is an instance of Series
    fig = ax.get_figure()
    fig.savefig(path)
    plt.close()

def extract_no_change_colums(df:t.Any, path:str) -> t.List[str]:
    if Path(path).is_file():
        return []
    std = df.std()
    return std[std==0].index.tolist()

def interpolate(df):
    meandiff = df.sort_values('Score').rolling(window=2).mean().dropna()
    return df.append(meandiff)

def flat_distorsion(df, bins=50):
    groups = df.groupby(pd.cut(df['Score'], bins))
    max_size = groups.size().max()
    dfs = []
    for idx, group in groups:
        if len(group) > 0:
            dfs.append(group.sample(max_size-len(group), replace=True))
    flat_df = pd.concat(dfs).append(df)
    return flat_df

@skip_if(
    lambda *args: Path(args[1]).is_file(),
    lambda *args: pd.read_pickle(args[1]),
)
def get_corr_mtrx(df:t.Any, path:str) -> None:
    corr = df.corr().abs()
    corr.to_pickle(path)
    return corr

@skip_if(
    lambda *args: Path(args[1]).is_file(),
)
def save_heatmap(df:t.Any, path:str) -> None:
    mask = np.zeros_like(df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        df,
        mask=mask,
    )
    f.savefig(path)


def load(path:str) -> t.Any:
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

@skip_if(
    lambda *args: Path(args[1]).is_file(),
    lambda *args: load(args[1]),
)
def dump(obj:t.Any, path:str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def get_ignore_columns(df:t.Any, threshold:float) -> t.Set[str]:
    if len(df.columns) == 0:
        return set()

    mask = np.ones_like(df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = False
    filtered = df.where(mask)
    ignore_columns:t.Set[str] = set()
    for c in filtered.columns:
        a = filtered[c][(filtered[c] > threshold ) & (filtered[c] < 1 )]
        ignore_columns = ignore_columns | set(a.index)
    return ignore_columns
