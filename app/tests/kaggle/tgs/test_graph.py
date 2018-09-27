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
        labeled_batch_size=32,
        no_labeled_batch_size=4,
        model_type='HybridUNet',
        val_split_size=0.2,
        feature_size=4,
        depth=4,
        patience=20,
        reduce_lr_patience=6,
        base_size=10,
        ema_decay=0.5,
        consistency=0.2,
        consistency_rampup=30,
        parallel=1,
        top_num=1,
    )

    with Client('dask-scheduler:8786') as c:
        try:
            result = g.output.compute()
        finally:
            c.restart()
