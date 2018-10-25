from pathlib import Path
from aplf.tellus.data import load_train_df, get_train_row, TellusDataset, kfold, ChunkSampler, get_test_row, load_test_df
from torch.utils.data import DataLoader
import pandas as pd
import pytest

from cytoolz.curried import keymap, filter, pipe, merge, map, compose, concatv, first, take, concat
from torch.utils.data import Subset
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from aplf import config


def test_get_train_row():
    rows = get_train_row(
        base_path='/store/tellus/train',
        label_dir="positive",
        label=1
    )
    for r in rows:
        assert Path(rows[0]['palser_after']).name == Path(
            rows[0]['palser_before']).name
        assert Path(rows[0]['palser_before']).name == Path(
            rows[0]['landsat_after']).name
        assert Path(rows[0]['landsat_after']).name == Path(
            rows[0]['landsat_before']).name

    assert len(rows) == 1530


def test_get_test_row():
    rows = get_test_row(
        base_path='/store/tellus/test',
    )
    for r in rows:
        assert Path(rows[0]['palser_after']).name == Path(
            rows[0]['palser_before']).name

    assert len(rows) == 133520


def test_test_dataset():
    output = load_test_df(
        dataset_dir='/store/tellus/test',
        output='/store/tmp/test.pqt'
    )
    df = pd.read_parquet(output)
    dataset = TellusDataset(
        df=df,
        has_y=False,
    )
    assert len(dataset[0]) == 3


def test_train_dataset():
    output = load_train_df(
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


@pytest.mark.parametrize("idx", range(1524, 1536))
def test_aug(idx):
    output = load_train_df(
        dataset_dir='/store/tellus/train',
        output='/store/tellus/train.pqt'
    )
    df = pd.read_parquet(output)

    dataset = TellusDataset(
        df=df,
        has_y=True,
    )

    writer = SummaryWriter(f'{config["TENSORBORAD_LOG_DIR"]}/test')
    writer.add_image(
        f"palser/{dataset[idx]['id']}/{dataset[idx]['label']}",
        vutils.make_grid(
            pipe(range(1),
                 map(lambda x: dataset[idx]),
                 map(lambda x: [
                     x['palser_before'],
                     x['palser_after'],
                 ]),
                 concat,
                 list)
        ),
    )

    writer.add_image(
        f"landsat/{dataset[idx]['id']}/{dataset[idx]['label']}",
        vutils.make_grid(
            pipe(range(1),
                 map(lambda x: dataset[idx]),
                 map(lambda x: [
                     x['landsat_before'],
                     x['landsat_after']
                 ]),
                 concat,
                 list)
        ),
    )


@pytest.mark.parametrize("idx", range(1520, 1536))
def test_diff(idx):
    output = load_train_df(
        dataset_dir='/store/tellus/train',
        output='/store/tellus/train.pqt'
    )
    df = pd.read_parquet(output)

    dataset = TellusDataset(
        df=df,
        has_y=True,
    )

    writer = SummaryWriter(f'{config["TENSORBORAD_LOG_DIR"]}/diff')
    writer.add_image(
        f"palser/{idx}/{dataset[idx]['label']}",
        vutils.make_grid(
            pipe(range(1),
                 map(lambda x: dataset[idx]),
                 map(lambda x: [
                     x['palser_after'] - x['palser_before'],
                 ]),
                 concat,
                 list)
        ),
    )

    writer.add_image(
        f"landsat/{idx}/{dataset[idx]['label']}",
        vutils.make_grid(
            pipe(range(1),
                 map(lambda x: dataset[idx]),
                 map(lambda x: [
                     x['landsat_after'] - x['landsat_before']
                 ]),
                 concat,
                 list)
        ),
    )


@pytest.mark.parametrize("idx", range(1520, 1536))
def test_sum(idx):
    output = load_train_df(
        dataset_dir='/store/tellus/train',
        output='/store/tellus/train.pqt'
    )
    df = pd.read_parquet(output)

    dataset = TellusDataset(
        df=df,
        has_y=True,
    )

    writer = SummaryWriter(f'{config["TENSORBORAD_LOG_DIR"]}/sum')
    writer.add_image(
        f"palser/{idx}/{dataset[idx]['label']}",
        vutils.make_grid(
            pipe(range(1),
                 map(lambda x: dataset[idx]),
                 map(lambda x: [
                     x['palser_after'] + x['palser_before'],
                 ]),
                 concat,
                 list)
        ),
    )

    writer.add_image(
        f"landsat/{idx}/{dataset[idx]['label']}",
        vutils.make_grid(
            pipe(range(1),
                 map(lambda x: dataset[idx]),
                 map(lambda x: [
                     x['landsat_after'] + x['landsat_before']
                 ]),
                 concat,
                 list)
        ),
    )
