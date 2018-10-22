from aplf.tellus.dataset import load_dataset_df, get_row, TellusDataset, to_pn_set
import pandas as pd
from torch.utils.data import Subset


def test_get_row():
    rows = get_row(
        base_path='/store/tellus/train',
        sat="LANDSAT",
        label_dir="positive",
        label=True
    )
    assert len(rows) == 1530



def test_dataset():
    output = load_dataset_df(
        dataset_dir='/store/tellus/train',
        output='/store/tmp/train.pqt'
    )
    df = pd.read_parquet(output)
    dataset = TellusDataset(
        df=df,
        has_y=True,
    )
    assert len(dataset[0]) == 4

def test_pn():
    output = load_dataset_df(
        dataset_dir='/store/tellus/train',
        output='/store/tmp/train.pqt'
    )
    df = pd.read_parquet(output)
    dataset = TellusDataset(
        df=df,
        has_y=True,
    )
    subset = Subset(
        dataset,
        list(range(1500, 1600))
    )
    pos_set, neg_set = to_pn_set(subset)
    assert len(pos_set) + len(neg_set) == 100
