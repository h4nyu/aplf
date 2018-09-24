from distributed import Client
from aplf.kaggle.tgs.graph import Graph
import uuid
from datetime import datetime


def test_graph():
    g = Graph(
        id=f"{datetime.now().isoformat()}",
        dataset_dir='/store/kaggle/tgs',
        output_dir='/store/kaggle/tgs/output',
        epochs=200,
        labeled_batch_size=16,
        no_labeled_batch_size=16,
        val_split_size=0.20,
        feature_size=16,
        depth=3,
        patience=20,
        base_size=10,
        ema_decay=0.1,
        consistency=1,
        consistency_rampup=0,
        parallel=3,
        top_num=2,
    )

    with Client('dask-scheduler:8786') as c:
        try:
            result = g.output.compute()
        finally:
            c.restart()
