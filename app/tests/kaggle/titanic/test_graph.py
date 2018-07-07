from distributed import Client
from aplf.kaggle.titanic.graph import(
    train_x,
    train_y,
    train_dataset,
    train_result,
    loss_plot,
    train_df
)


def test_graph():
    with Client('dask-scheduler:8786') as c:
        try:
            target = loss_plot
            target.visualize('/data/titanic/graph.svg')
            result = target.compute()
            print(result)
        finally:
            c.restart()
