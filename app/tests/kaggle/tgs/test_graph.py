from distributed import Client
from aplf.kaggle.tgs.graph import Graph
import uuid
from datetime import datetime


def test_graph():
    g = Graph(
        id="2",
        dataset_dir='/store/kaggle/tgs',
        output_dir='/store/kaggle/tgs/output',
        epochs=400,
        labeled_batch_size=32,
        no_labeled_batch_size=1,
        model_type='RUNet',
        val_split_size=0.2,
        feature_size=8,
        depth=3,
        patience=20,
        base_size=10,
        ema_decay=0.99,
        consistency=0,
        consistency_rampup=50,
        cyclic_period=5,
        switch_epoch=100,
        milestones=[(0, 1)],
        parallel=4,
        top_num=2,
    )

    with Client('dask-scheduler:8786') as c:
        try:
            result = g.output.compute()
        finally:
            c.restart()
