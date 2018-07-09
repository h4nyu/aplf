from distributed import Client
from aplf.kaggle.titanic.graph import(
    train_x,
    train_y,
    train_dataset,
    train_result,
    train_df,
    predict_result,
    test_dataset,
    submission_df,
    save_submission,
)


def test_graph():
    with Client('dask-scheduler:8786') as c:
        try:
            target = save_submission
            target.visualize('/store/kaggle/titanic/graph.svg')
            result = target.compute()
            print(result)
        finally:
            c.restart()
