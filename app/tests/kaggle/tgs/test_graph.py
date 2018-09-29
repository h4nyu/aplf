from distributed import Client
from aplf.kaggle.tgs.graph import Graph
import uuid
from datetime import datetime


def test_graph():
    g = Graph(
        id="feature-8-depth-4-unet-ema-0.99-consistency-0.2-bs-24-rampup-30-switch-100-cyclic-10-epoch-400-milestones-50-99",
        dataset_dir='/store/kaggle/tgs',
        output_dir='/store/kaggle/tgs/output',
        epochs=400,
        labeled_batch_size=32,
        no_labeled_batch_size=32,
        model_type='UNet',
        val_split_size=0.2,
        feature_size=12,
        depth=4,
        patience=20,
        base_size=10,
        ema_decay=0.99,
        consistency=0.2,
        consistency_rampup=100,
        cyclic_period=10,
        switch_epoch=100,
        milestones=[(0, 0.01), (50, 0.005), (100, 0.002)],
        parallel=1,
        top_num=1,
    )

    with Client('dask-scheduler:8786') as c:
        try:
            result = g.output.compute()
        finally:
            c.restart()
