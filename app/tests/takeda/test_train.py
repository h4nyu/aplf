from aplf.takeda.train.nn import regular_loss
from aplf.takeda.models import Model
from torch.utils.data import Dataset
from torch import tensor, rand, Tensor, randn
import pytest
import typing as t
import torchvision


class DatasetMock(Dataset):
    def __len__(self) -> int:
        return 30

    def __getitem__(self, index: int) -> t.Tuple[float, Tensor]:
        return rand(10), randn(1)



def test_regular_loss() -> None:
    pres = randn(4, 2)
    loss = regular_loss(pres, -0.5, 5)
    print(loss)
