from distributed import Client
from aplf.kaggle.tgs.graph import Graph
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
        id="17",
        epochs=400,
        labeled_batch_size=32,
        no_labeled_batch_size=16,
        model_type='HUNet',
        feature_size=16,
        depth=3,
        patience=30,
        base_size=10,
        ema_decay=0.1,
        consistency=1,
        consistency_rampup=20,
        cyclic_period=5,
        switch_epoch=30,
        milestones=[(0, 1)],
        parallel=2,
        top_num=2,
        erase_num=3,
    )

    with Client('dask-scheduler:8786') as c:
        try:
            result = g.output.compute()
        finally:
            c.restart()
