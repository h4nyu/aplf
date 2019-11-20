from pathlib import Path
from .data import read_table, Table, kfold, resize_all
from .train import train_cv

def main() -> None:
    size = (300, 300)
    currnet_dir = Path("/store/tmp")
    currnet_dir.mkdir(exist_ok=True)
    resize_all(
        Path('/store/t3tsc/train_images'),
        Path('/store/tmp/downsampled_train'),
        size=size,
        pattern='*.jpg'
    )
    resize_all(
        Path('/store/t3tsc/train_annotations'),
        Path('/store/tmp/downsampled_annotations'),
        size=size,
        pattern='*.png'
    )
    table = read_table(
        x_dir=Path('/store/t3tsc/downsampled_train'),
        y_dir=Path('/store/tmp/downsampled_annotations'),
    )

    index_pairs = kfold(
        table,
        n_splits=4
    )
    for train_indices, test_indices in index_pairs:
        train_cv(
            table,
            train_indices,
            test_indices,
            n_epochs=100,
        )
    #
