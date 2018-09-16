from distributed import Client
from aplf.kaggle.tgs.graph import Graph
import uuid
from datetime import datetime


def test_graph():
    g = Graph(
        id=f"{datetime.now().isoformat()}",
        dataset_dir='/store/kaggle/tgs',
        output_dir='/store/kaggle/tgs/output',
        batch_size=32,
        epochs=1000,
        val_split_size=0.1,
        feature_size=24,
        patience=20,
        base_size=10,
        devices=['cuda', 'cuda', 'cuda'],
        top_num=2,
    )

    with Client('dask-scheduler:8786') as c:
        try:
            result = g.output.compute()
            print(result)
        finally:
            c.restart()
