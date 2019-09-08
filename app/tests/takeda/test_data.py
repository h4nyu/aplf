from aplf.takeda.data import (
    TakedaDataset, 
    kfold, 
    save_submit,
    csv_to_pkl,
    extract_col_type,
)
from torch.utils.data import Dataset
import typing as t
import pandas as pd
from torch import tensor, Tensor
import torchvision


def test_csv_to_pkl() -> None:
    path = csv_to_pkl(
        '/store/takeda/train.csv',
        '/store/takeda/train.pkl',
    )
    assert path == '/store/takeda/train.pkl'




def test_dataset() -> None:
    df = pd.DataFrame({
        'Score': [0., 1.],
        'col1': [10., 20.],
        'col2': [0., 1.],
    }, index=[0, 1])
    dataset = TakedaDataset(df)
    assert len(dataset) == 2
    print(dataset[0])

    assert 0. == dataset[0][1]
    assert 1. == dataset[1][1]



def test_kfold() -> None:
    class DatasetMock(Dataset):
        def __len__(self) -> int:
            return 10

        def __getitem__(self, index: int) -> t.Tuple[float, Tensor]:
            return rand(10), rand(1)
    dataset = DatasetMock()
    res = kfold(dataset, n_splits=3)
    assert len(res) == 3


def test_save_submit() -> None:
    df = pd.DataFrame({
        'Score': [0., 1., 2.5, 10],
        'col1': [0., 1., 0., 1, ],
        'col2': [0., 1., 1.5, 20],
        'col3': [0., 1., 3, 0.5],
    }, index=[0, 1, 2, 3])
    preds = [11, 22, 33, 44]
    df = save_submit(df, preds, '/store/submit.csv')
