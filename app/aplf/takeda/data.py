from logging import getLogger
from pathlib import Path
from torch.utils.data import Dataset
import typing as t
import pandas as pd
import numpy as np
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

def add_noise(df, unique_threshold=2):
    indices = []
    for idx in range(len(df.columns)):
        print(df.iloc[:, idx].unique())
        print(idx)
        if len(df.iloc[:, idx].unique()) > unique_threshold:
            indices.append(idx)




class TakedaDataset(Dataset):
    def __init__(self, df:t.Any) -> None:
        self.y = df['Score'].values
        self.x = df.drop('Score', axis=1).values
        self.float_indices = self.__get_float_indices()

    def __get_float_indices(self) -> t.List[int]:
        indices = []
        for i in range(self.x.shape[1]):
            if len(np.unique(self.x[:, i])) > 10:
                indices.append(i)
        return indices

    def __get_rand_tensor(self, indices:t.List[int]) -> t.Any:
        base = np.zeros(self.x.shape[1])
        rand = np.random.randn(len(self.float_indices)) * 0.05
        for i, j in enumerate(self.float_indices):
            base[j] = rand[i]
        return base


    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx:int) -> t.Tuple[Tensor, Tensor]:
        if np.random.randn() > 0:
            rand_t = self.__get_rand_tensor(self.float_indices)
            x = self.x[idx] + rand_t
        else:
            x = self.x[idx]
        y = self.y[idx]
        return tensor(x, dtype=float32), tensor(y, dtype=float32)

