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
        self.series_y = df['Score']
        self.df_x = df.drop('Score', axis=1)

    def __len__(self) -> int:
        return len(self.series_y)

    def __getitem__(self, idx:int) -> t.Tuple[Tensor, Tensor]:
        x = self.df_x.iloc[idx]
        y = self.series_y[idx]
        return tensor(x), tensor(y)
