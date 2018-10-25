import h5py
import numpy as np
from cytoolz.curried import keymap, filter, pipe, merge, map, reduce
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from skimage import io
import torch
from dask import delayed
import pandas as pd
from aplf import config
from aplf.utils import skip_if_exists
from .preprocess import rl_enc
from .metric import iou


@skip_if_exists('out_path')
def predict(model_paths,
            dataset,
            out_path,
            log_dir,
            log_interval=100,
            ):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda")

    models = pipe(model_paths,
                  map(torch.load),
                  map(lambda x: x.eval().to(device)),
                  list)

    rows = []

    with torch.no_grad():
        for sample in loader:
            sample_id = sample['id'][0]
            palser_before = sample['palser_before'].to(device)
            palser_after = sample['palser_after'].to(device)

            normal_outputs = pipe(
                models,
                map(lambda x: x(
                    palser_before,
                    palser_after
                )[0]),
                list,
            )

            fliped_outputs = pipe(
                models,
                map(lambda x: x(
                    palser_before.flip([3]),
                    palser_after.flip([3]),
                )[0]),
                list,
            )
            output = pipe(
                [*normal_outputs, *fliped_outputs],
                map(lambda x: x.softmax(dim=1)),
                reduce(lambda x, y: x + y / 2),
                lambda x: F.softmax(x, dim=1),
                lambda x: x.argmax(dim=1)[0]
            )
            row = {
                'id': sample_id,
                'label': int(output)
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(out_path, sep='\t', header=False, index=False)
        return out_path
