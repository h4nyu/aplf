from distributed import Client
import aplf.kaggle.titanic.graph as g


def test_graph():
    with Client('dask-scheduler:8786') as c:
        try:
            target = g.train_df
            target.visualize('/store/kaggle/titanic/graph.svg')
            result = target.compute()
            print(result)
        finally:
            c.restart()
