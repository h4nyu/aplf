from distributed import Client
from aplf.kaggle.tgs.graph import Graph, SelectionGraph
import uuid
from datetime import datetime


base_param = {
    "dataset_dir": '/store/kaggle/tgs',
    "output_dir": '/store/kaggle/tgs/output',
    "val_split_size": 0.2,
}


def test_graph():
    g = Graph(
        **base_param,
        id="8",
        epochs=800,
        labeled_batch_size=32,
        no_labeled_batch_size=1,
        model_type='EUNet',
        feature_size=16,
        depth=3,
        patience=30,
        base_size=10,
        ema_decay=1,
        consistency=0,
        consistency_rampup=30,
        cyclic_period=5,
        switch_epoch=100,
        milestones=[(0, 1)],
        parallel=1,
        top_num=1,
    )

    with Client('dask-scheduler:8786') as c:
        try:
            result = g.output.compute()
        finally:
            c.restart()


def test_selection():
    train_config = {
        "epochs": 1,
        "labeled_batch_size": 32,
        "no_labeled_batch_size": 1,
        "model_type": 'DUNet',
        "feature_size": 16,
        "depth": 3,
        "ema_decay": 1,
        "consistency": 0,
        "cyclic_period": 5,
        "consistency_rampup": 30,
        "switch_epoch": 100,
        "milestones": [(0, 1)],
    }
    g = SelectionGraph(
        **base_param,
        id="9",
        parallel=1,
        random_state=0,
        n_generation=3,
        bins=4,
        train_config=train_config,
    )

    with Client('dask-scheduler:8786') as c:
        try:
            result = g.output.compute()
            print(result)
        finally:
            c.restart()
