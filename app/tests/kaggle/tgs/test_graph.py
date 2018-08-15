from distributed import Client
from aplf.kaggle.tgs.graph import Graph


def test_dataset():
    g = Graph(
        dataset_dir='/store/kaggle/tgs',
        output_dir='/store/kaggle/tgs/output',
        batch_size=32,
        epochs=30,
        val_split_size=0.1,
        patience=20,
        parallel=5
    )

    with Client('dask-scheduler:8786') as c:
        try:
            result = g.output.compute()
            print(result[0])
        finally:
            c.restart()
