from aplf.takeda.train import train_epoch
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

def test_train_epoch() -> None:
    dataset = DatasetMock()
    model = Model(
        size_in=10,
    )

    out = train_epoch(
        model,
        dataset,
        batch_size=10,
    )
