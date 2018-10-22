from aplf.tellus.data import load_dataset_df, get_row, TellusDataset, kfold, ChunkSampler
from torch.utils.data import DataLoader
import pandas as pd

from cytoolz.curried import keymap, filter, pipe, merge, map, compose, concatv, first, take
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


def test_kfold():
    output = load_dataset_df(
        dataset_dir='/store/tellus/train',
        output='/store/tmp/train.pqt'
    )
    df = pd.read_parquet(output)
    sets = kfold(df, n_splits=10)
    for s in sets:
        assert pipe(
            s['train_pos'],
            take(100),
            map(lambda x: x['label']),
            filter(lambda x: x == 0),
            list,
            len
        ) == 0
        assert pipe(
            s['val_pos'],
            take(100),
            map(lambda x: x['label']),
            filter(lambda x: x == 0),
            list,
            len
        ) == 0
        assert pipe(
            s['train_neg'],
            take(100),
            map(lambda x: x['label']),
            filter(lambda x: x == 1),
            list,
            len
        ) == 0
        assert pipe(
            s['val_neg'],
            take(100),
            map(lambda x: x['label']),
            filter(lambda x: x == 1),
            list,
            len
        ) == 0
        assert len(s) == 4


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
