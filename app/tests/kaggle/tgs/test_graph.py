from distributed import Client
from aplf.kaggle.tgs.graph import Graph
import uuid
from datetime import datetime


def test_graph():
    g = Graph(
        id=f"{datetime.now().isoformat()}",
        dataset_dir='/store/kaggle/tgs',
        output_dir='/store/kaggle/tgs/output',
        epochs=1200,
        labeled_batch_size=32,
        no_labeled_batch_size=64,
        val_split_size=0.20,
        feature_size=16,
        depth=3,
        patience=20,
        base_size=10,
        ema_decay=0.5,
        consistency=0.5,
        consistency_rampup=30,
        parallel=4,
        top_num=2,
    )

    with Client('dask-scheduler:8786') as c:
        try:
            result = g.output.compute()
            print(result)
        finally:
            c.restart()
