from aplf.takeda.data import read_csv, TakedaDataset
from torch.utils.data import Dataset
import pandas as pd

def test_read_csv() -> None:
    df = read_csv('/store/takeda/train.csv')
    assert len(df.columns) == 3806

def test_dataset() -> None:
    df = pd.DataFrame({
        'Score': [0, 0],
        'col1': [0, 0],
    }, index=[0, 1])
    dataset = TakedaDataset(df)
