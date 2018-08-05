from distributed import Client
from aplf.kaggle.tgs_salt.graph import Graph


def test_dataset():
    g = Graph(
        dataset_dir='/store/kaggle/tgs-salt',
        model_path='/store/kaggle/tgs-salt/model.pt',
    )

    with Client('dask-scheduler:8786') as c:

        try:
            result = c.compute(
                g.trained,
                sync=True
            )
            print(result)
        finally:
            c.restart()
