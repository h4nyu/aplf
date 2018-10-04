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
        id="15",
        epochs=400,
        labeled_batch_size=32,
        no_labeled_batch_size=1,
        model_type='HUNet',
        feature_size=32,
        depth=3,
        patience=30,
        base_size=10,
        ema_decay=0,
        consistency=0,
        consistency_rampup=0,
        cyclic_period=5,
        switch_epoch=300,
        milestones=[(0, 1)],
        parallel=5,
        top_num=1,
        erase_num=1,
    )

    with Client('dask-scheduler:8786') as c:
        try:
            result = g.output.compute()
        finally:
            c.restart()
