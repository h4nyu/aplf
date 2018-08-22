from distributed import Client
from aplf.kaggle.tgs.graph import Graph


def test_graph():
    g = Graph(
        dataset_dir='/store/kaggle/tgs',
        output_dir='/store/kaggle/tgs/output',
        batch_size=32,
        epochs=200,
        val_split_size=0.3,
        patience=15,
        base_size=10,
        parallel=1,
        top_num=1,
    )

    with Client('dask-scheduler:8786') as c:
        try:
            result = g.output.compute()
            print(result)
        finally:
            c.restart()