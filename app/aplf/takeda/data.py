from logging import getLogger
from pathlib import Path
from torch.utils.data import Dataset
import typing as t
import pandas as pd
from torch import Tensor, tensor, float32
from typing_extensions import Protocol


logger = getLogger("takeda.data")
def read_csv(path: str) -> t.Any:
    df = pd.read_csv(path)
    df = df.set_index('ID')
    return df


class TakedaDataset(Dataset):
    def __init__(self, df:t.Any) -> None:
        self.y = df['Score'].values
        self.x = df.drop('Score', axis=1).values

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx:int) -> t.Tuple[Tensor, Tensor]:
        x = self.x[idx]
        y = self.y[idx]
        return tensor(x, dtype=float32), tensor(y, dtype=float32)
