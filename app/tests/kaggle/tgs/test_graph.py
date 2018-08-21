from distributed import Client
from aplf.kaggle.tgs.graph import Graph


def test_dataset():
    g = Graph(
        dataset_dir='/store/kaggle/tgs',
        output_dir='/store/kaggle/tgs/output',
        batch_size=32,
        epochs=200,
        val_split_size=0.3,
        patience=10,
        base_size=10,
        parallel=2,
        top_num=1,
    )

    with Client('dask-scheduler:8786') as c:
        try:
            result = g.output.compute()
            print(result)
        finally:
            c.restart()
