from aplf.t3tsc.data import read_table, Dataset, resize
from pathlib import Path
from aplf.utils import Timer


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
