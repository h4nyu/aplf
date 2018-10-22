from aplf.tellus.dataset import load_dataset_df, get_row, TellusDataset, to_pn_set, ChunkSampler
from torch.utils.data import DataLoader
import pandas as pd

from cytoolz.curried import keymap, filter, pipe, merge, map, compose, concatv, first
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

def test_esampler():

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

    s = ChunkSampler(
        epoch_size=10,
        len_indices=len(subset),
        shuffle=True,
    )

    train_loader = DataLoader(
        subset,
        sampler=s,
        batch_size=2,
        pin_memory=True,
    )
    for i in range(11):
        samples = pipe(
            train_loader,
            map(lambda x: x['id']),
            list
        )
        assert len(samples) == 5


