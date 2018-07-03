from distributed import Client
from aplf.kaggle.titanic.graph import train_dataset, train_result, loss_plot


def test_graph():
    with Client('dask-scheduler:8786') as c:
        try:
            target = loss_plot
            target.visualize('/data/titanic/graph.svg')
            result = c.compute(target, sync=True)
            #  print(result.input_len)
            print(result[1])
        finally:
            c.restart()
