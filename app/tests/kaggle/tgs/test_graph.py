from distributed import Client
from aplf.kaggle.tgs.graph import Graph
import uuid
from datetime import datetime


def test_graph():
    g = Graph(
        id=f"{datetime.now().isoformat()}",
        dataset_dir='/store/kaggle/tgs',
        output_dir='/store/kaggle/tgs/output',
        epochs=1500,
        labeled_batch_size=14,
        no_labeled_batch_size=2,
        val_split_size=0.2,
        feature_size=6,
        depth=3,
        patience=20,
        reduce_lr_patience=6,
        base_size=10,
        ema_decay=0.5,
        consistency=0.1,
        consistency_rampup=20,
        parallel=4,
        top_num=2,
    )

    with Client('dask-scheduler:8786') as c:
        try:
            result = g.output.compute()
        finally:
            c.restart()
