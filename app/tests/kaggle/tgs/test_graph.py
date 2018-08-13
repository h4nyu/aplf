from distributed import Client
from aplf.kaggle.tgs.graph import Graph


def test_dataset():
    g = Graph(
        dataset_dir='/store/kaggle/tgs',
        model_path='/store/kaggle/tgs/model.pt',
    )

    with Client('dask-scheduler:8786') as c:
        try:
            result = g.output.compute()
            print(result)
        finally:
            c.restart()
