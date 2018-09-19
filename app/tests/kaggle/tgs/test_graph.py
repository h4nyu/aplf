from distributed import Client
from aplf.kaggle.tgs.graph import Graph
import uuid
from datetime import datetime


def test_graph():
    g = Graph(
        id=f"{datetime.now().isoformat()}",
        dataset_dir='/store/kaggle/tgs',
        output_dir='/store/kaggle/tgs/output',
        epochs=800,
        labeled_batch_size=32,
        no_labeled_batch_size=64,
        val_split_size=0.20,
        feature_size=64,
        depth=4,
        patience=20,
        base_size=10,
        ema_decay=0.5,
        consistency=1,
        consistency_rampup=30,
        parallel=1,
        top_num=1,
    )

    with Client('dask-scheduler:8786') as c:
        try:
            result = g.output.compute()
            print(result)
        finally:
            c.restart()
