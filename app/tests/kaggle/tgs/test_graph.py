from distributed import Client
from aplf.kaggle.tgs.graph import Graph


def test_dataset():
    g = Graph(
        dataset_dir='/store/kaggle/tgs',
        output_dir='/store/kaggle/tgs/output',
        batch_size=64,
        epochs=40,
        val_split_size=0.1,
        patience=20,
    )

    with Client('dask-scheduler:8786') as c:
        try:
            result = g.output.compute()
            print(result)
        finally:
            c.restart()
