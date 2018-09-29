from distributed import Client
from aplf.kaggle.tgs.graph import Graph
import uuid
from datetime import datetime


def test_graph():
    g = Graph(
        id="feature-8-depth-4-unet-ema-0.99-consistency-1-lbs-32-nlbs-32-rampup-70-switch-70-cyclic-10-epoch-400-milestones-50-0.5-100-0.2-noise-0.3",
        dataset_dir='/store/kaggle/tgs',
        output_dir='/store/kaggle/tgs/output',
        epochs=400,
        labeled_batch_size=24,
        no_labeled_batch_size=24,
        model_type='UNet',
        val_split_size=0.2,
        feature_size=8,
        depth=4,
        patience=20,
        base_size=10,
        ema_decay=0.99,
        consistency=1,
        consistency_rampup=70,
        cyclic_period=10,
        switch_epoch=70,
        milestones=[(0, 1), (50, 0.5), (100, 0.2)],
        parallel=1,
        top_num=1,
    )

    with Client('dask-scheduler:8786') as c:
        try:
            result = g.output.compute()
        finally:
            c.restart()
