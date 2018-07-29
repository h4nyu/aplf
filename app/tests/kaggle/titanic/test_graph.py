from distributed import Client
import aplf.kaggle.titanic.graph as g


def test_graph():
    with Client('dask-scheduler:8786') as c:
        try:
            target = g.preprocessed_train_df
            result = target.compute()
            print(result['SexCode'])
        finally:
            c.restart()
