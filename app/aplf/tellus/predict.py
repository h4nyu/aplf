import glob
import h5py
import numpy as np
from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, concat
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from dask import delayed
import pandas as pd
from aplf import config
from aplf.utils import skip_if_exists
import os
import torch
from .preprocess import rl_enc
from .metric import iou
import json


def get_threshold(path):
    with open(path, 'r') as f:
        return json.load(f)['threshold']


@skip_if_exists('out_path')
def predict(model_paths,
            dataset,
            out_path,
            batch_size=512,
            ):

    device = torch.device("cuda")
    models = pipe(
        model_paths,
        map(torch.load),
        map(lambda x: x.eval().to(device)),
        list
    )
    thresholds = pipe(
        model_paths,
        map(lambda x: f"{x}.json"),
        map(get_threshold),
        list
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    rows = []

    y_preds = []
    y_ids = []
    with torch.no_grad():
        for sample in loader:
            ids = sample['id']
            palser_x = sample['palsar'].to(device)

            normal_outputs = pipe(
                zip(models, thresholds),
                map(lambda x: x[0](palser_x)[0].softmax(dim=1)[:, 1] > x[1]),
                map(lambda x: x.float()),
                list,
            )
            output = pipe(
                [*normal_outputs],
                reduce(lambda x, y: x+y),
                lambda x: x >= 1,
            )
            y_ids += ids
            y_preds += output.cpu().detach().tolist()

        rows = pipe(
            zip(y_ids, y_preds),
            map(lambda x: {'id': x[0], 'lable': x[1]}),
            list
        )
        df = pd.DataFrame(rows)
        df.to_csv(out_path, sep='\t', header=False, index=False)
        return out_path
