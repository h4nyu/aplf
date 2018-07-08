from distributed import Client
from aplf.kaggle.titanic.graph import(
    train_x,
    train_y,
    train_dataset,
    train_result,
    loss_plot,
    train_df
)


def test_dataset():
    #  with Client('dask-scheduler:8786') as c:
    result = train_dataset.compute()
    print(result[0], result[1])
