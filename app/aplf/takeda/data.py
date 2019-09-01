from logging import getLogger
from pathlib import Path
from torch.utils.data import Dataset
import typing as t
import pandas as pd
from torch import Tensor, tensor, float32
from typing_extensions import Protocol
from sklearn.model_selection import KFold


logger = getLogger("takeda.data")
def read_csv(path: str) -> t.Any:
    df = pd.read_csv(path)
    df = df.set_index('ID')
    return df

def kfold(
    dataset:Dataset, 
    n_splits:int, 
    random_state:int=0
) -> t.List[t.Tuple[t.Sequence[int], t.Sequence[int]]]:
    kf = KFold(n_splits, random_state=random_state, shuffle=True)
    return list(kf.split(dataset))

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
