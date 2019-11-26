import typing as t
from aplf.t3tsc.data import read_table, Dataset, resize, get_iou, to_binary, train_aug
import pytest
from pathlib import Path
from aplf.utils import Timer
from torch import randn, empty, tensor
import Augmentor
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
        Path('/store/tmp/downsampled_train'),
        Path('/store/tmp/downsampled_annotations'),
    )
    dset = Dataset(table)
    x, y = dset[0]
    print(y)
    assert len(np.unique(y)) == 12



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

def test_horizontal_flip() -> None:
    img = np.array([[[1,  0],
                    [1,  0],
                    [1,  0]]])
    out_img = horizontal_flip(img)
    print(out_img.shape)

    expected = np.array([[[0,  1],
                         [0,  1],
                         [0,  1]]])
    print(out_img)
    assert np.abs(out_img - expected).sum() == 0



def test_train_aug() -> None:
    mask = np.array([[.0,  .1],
                     [.1,  .0]])

    img0 = np.array([[.0,  .1],
                     [.2,  .3]])
    img1 = np.array([[.0,  .1],
                     [.2,  .3]])

    hh, hv, mask = train_aug([img0, img1, mask], probability=1)
    print(hh)
    print(hv)
    print(mask)
