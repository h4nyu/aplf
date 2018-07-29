from distributed import Client
import aplf.kaggle.titanic.graph as g


def test_graph():
    with Client('dask-scheduler:8786') as c:
        try:
            target = g.predict_result
            result = target.compute()
            print(result)
        finally:
            c.restart()
