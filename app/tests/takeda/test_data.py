from aplf.takeda.data import read_csv, TakedaDataset, kfold
from torch.utils.data import Dataset
import typing as t
import pandas as pd
from torch import tensor, Tensor
import torchvision


def test_read_csv() -> None:
    df = read_csv('/store/takeda/train.csv')
    assert len(df.columns) == 3806

def test_dataset() -> None:
    df = pd.DataFrame({
        'Score': [0., 1.],
        'col1': [0., 1.],
        'col2': [0., 1.],
    }, index=[0, 1])
    dataset = TakedaDataset(df)
    assert len(dataset) == 2
    print(dataset[0])

    assert  all(tensor([0., 0.]) == dataset[0][0])
    assert  0. == dataset[0][1]
    assert  all(tensor([1., 1.]) == dataset[1][0])
    assert  1. == dataset[1][1]



def test_kfold() -> None:
    class DatasetMock(Dataset):
        def __len__(self) -> int:
            return 10

        def __getitem__(self, index: int) -> t.Tuple[float, Tensor]:
            return rand(10), randn(1)
    dataset = DatasetMock()
    res =kfold(dataset, n_splits=3)
    assert len(res) == 3
