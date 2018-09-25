from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk, curry, merge
from distributed import Client
from aplf.kaggle.tgs.graph import SubmissionGraph
import uuid
from datetime import datetime


def test_graph():
    output_dir='/store/kaggle/tgs/output'
    model_paths = pipe(
        [
            '3f59adaf-5f0b-4aa7-ab27-61f1f521f68c',
            'b8969daa-34b3-4a6b-948f-b0283c2e004d',
        ],
        map(lambda x: f"{output_dir}/model_{x}.pt"),
        list
    )
    print(model_paths)

    g = SubmissionGraph(
        id=f"{datetime.now().isoformat()}",
        dataset_dir='/store/kaggle/tgs',
        output_dir=output_dir,
        model_paths=model_paths

    )

    with Client('dask-scheduler:8786') as c:
        try:
            result = g.output.compute()
        finally:
            c.restart()

