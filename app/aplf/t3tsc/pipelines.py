from pathlib import Path
from .data import read_table, Table, kfold, resize_all
from .train import train_cv
from logging import getLogger
from aplf.config import mlboard_url
from mlboard_client.writers import Writer


logger = getLogger('t3tsc')

def main(
    **params,
) -> None:
    writer = Writer(
        mlboard_url,
        't3tsc',
        params=params,
        logger=logger,
    )
    size = (256, 256)
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
        x_dir=Path('/store/tmp/downsampled_train'),
        y_dir=Path('/store/tmp/downsampled_annotations'),
    )

    index_pairs = kfold(
        table,
        n_splits=params['n_splits'],
    )
    for train_indices, test_indices in index_pairs:
        train_cv(
            table,
            train_indices,
            test_indices,
            n_epochs=params['n_epochs'],
            lr=params['lr'],
            momentum=params['momentum'],
            weight_decay=params['weight_decay'],
            writer=writer,
        )
