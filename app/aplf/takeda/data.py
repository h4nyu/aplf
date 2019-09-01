from pathlib import Path
import typing as t
import pandas as pd
from torch import Tensor, tensor


def read_csv(path: str) -> t.Any:
    df = pd.read_csv(path)
    df = df.set_index('ID')
    return df

class TakedaDataset:
    def __init__(self, df:t.Any) -> None:
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx:int) -> Tensor:
        row = self.df.loc[idx]
        return tensor(row)
