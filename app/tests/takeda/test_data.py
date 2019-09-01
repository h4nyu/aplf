from aplf.takeda.data import read_csv, TakedaDataset
from torch.utils.data import Dataset
import pandas as pd
from torch import tensor
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
    assert  all(tensor([0., 0.]) == dataset[0][0])
    assert  tensor(0.) == dataset[0][1]
    assert  all(tensor([1., 1.]) == dataset[1][0])
    assert  tensor(1.) == dataset[1][1]
