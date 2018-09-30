from distributed import Client
from aplf.kaggle.tgs.graph import Graph
import uuid
from datetime import datetime


def test_graph():
    g = Graph(
        #  id="feature-8-depth-3-unet-ema-0.99-consistency-1-lbs-32-nlbs-12-rampup-30-cyclic-10-epoch-400-milestones-50-0.5-100-0.2-noise-0.0-crossentorpy",
        id="test",
        dataset_dir='/store/kaggle/tgs',
        output_dir='/store/kaggle/tgs/output',
        epochs=400,
        labeled_batch_size=32,
        no_labeled_batch_size=1,
        model_type='UNet',
        val_split_size=0.2,
        feature_size=32,
        depth=3,
        patience=20,
        base_size=10,
        ema_decay=0.99,
        consistency=0,
        consistency_rampup=30,
        cyclic_period=5,
        switch_epoch=70,
        milestones=[(0, 1), (100, 0.8)],
        parallel=1,
        top_num=1,
    )

    with Client('dask-scheduler:8786') as c:
        try:
            result = g.output.compute()
        finally:
            c.restart()
