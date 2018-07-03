from distributed import Client
from aplf.kaggle.titanic.graph import train_dataset


def test_graph():
    with Client('dask-scheduler:8786') as c:
        try:
            target = train_dataset
            target.visualize('/data/titanic/graph.svg')
            result = c.compute(target, sync=True)
            print(result)
            print(result[0])
        finally:
            c.restart()
