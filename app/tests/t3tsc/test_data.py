import typing as t
from aplf.t3tsc.data import read_table, Dataset, resize, get_iou
import pytest
from pathlib import Path
from aplf.utils import Timer
from torch import randn, empty, tensor
import numpy as np


def test_read_table() -> None:
    table = read_table(
        Path('/store/t3tsc/train_images'),
        Path('/store/t3tsc/train_annotations'),
    )
    assert len(table) == 80


def test_resize() -> None:
    resize(
        Path('/store/t3tsc/train_images/train_hh_05.jpg'),
        Path('/tmp/test.jpg'),
    )

def test_dataset() -> None:
    table = read_table(
        Path('/store/t3tsc/train_images'),
        Path('/store/t3tsc/train_annotations'),
    )
    dset = Dataset(table)
    timer = Timer()
    with timer:
        for i in range(len(dset)):
            dset[i]
            print(i)
    print(timer.interval)
    with timer:
        for i in range(len(dset)):
            dset[i]
            print(i)
    print(timer.interval)

def test_dataset_get_item() -> None:
    table = read_table(
        Path('/store/t3tsc/downsampled_train'),
        Path('/store/t3tsc/downsampled_annotations'),
    )
    dset = Dataset(table)
    x, y = dset[0]
    assert len(np.unique(y)) == 13



@pytest.mark.parametrize("pred, label, expected", [
    (
        [[1, 1, 1],
         [1, 2, 1],
         [1, 1, 1]],
        [[1, 2, 1],
         [1, 2, 1],
         [1, 2, 1]],
        0.75
    )
])
def test_get_iou(pred:t.Any, label:t.Any, expected) -> None:
    pred = tensor(pred)
    label = tensor(label)
    score = get_iou(pred, label, classes=[1])
    assert score == expected
