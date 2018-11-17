from pathlib import Path
import dask
from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk, last
from sklearn.metrics import confusion_matrix
from dask import delayed
import random
import torchvision.utils as vutils
from pathlib import Path
from cytoolz import curry
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import torch.nn.functional as F
import pandas as pd
import numpy as np
from .. import models as mdl
from aplf.utils import EarlyStop
from aplf import config
from tensorboardX import SummaryWriter
from ..metric import iou
from os import path
from aplf.utils import skip_if_exists, dump_json
from aplf.losses import SSIM
from aplf.optimizers import Eve
from ..data import ChunkSampler, Augment, batch_aug
import uuid


def criterion(x, y):
    return nn.MSELoss(size_average=True)(x, y)


def validate(models,
             loader):
    models = pipe(
        models,
        map(lambda x: x.eval()),
        list
    )
    device = torch.device("cuda")
    sum_loss = 0
    epoch_len = len(loader)
    model_len = len(models)
    for sample in loader:
        with torch.no_grad():
            palsar = sample['palsar'].to(device)
            landsat = sample['landsat'].to(device)
            loss = pipe(
                models,
                map(lambda x: criterion(x(palsar, part='landsat'), landsat)),
                reduce(lambda x, y: (x+y)),
                lambda x: x/model_len
            )
            sum_loss += loss.item()
    mean_loss = sum_loss / epoch_len

    return {
        'ssim': mean_loss,
    }


def train_epoch(model,
                pos_loader,
                neg_loader,
                device,
                lr,
                landsat_weight,
                ):
    model = model.train()
    batch_len = len(pos_loader)

    optimizer = optim.Adam(
        model.landsat_enc.parameters(),
        amsgrad=True,
        lr=lr
    )
    sum_loss = 0
    for pos_sample, neg_sample in zip(pos_loader, neg_loader):
        palsar_x = torch.cat(
            [pos_sample['palsar'], neg_sample['palsar']],
            dim=0
        ).to(device)
        landsat_x = torch.cat(
            [pos_sample['landsat'], neg_sample['landsat']],
            dim=0
        ).to(device)

        loss = - landsat_weight * \
            criterion(model(palsar_x, part='landsat'), landsat_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    mean_loss = sum_loss / batch_len
    return model, {"ssim": mean_loss}


@skip_if_exists('model_dir')
def train(model_dir,
          sets,
          model_kwargs,
          epochs,
          batch_size,
          log_dir,
          landsat_weight,
          lr,
          num_ensamble,
          neg_scale,
          ):

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    models = pipe(
        range(num_ensamble),
        map(lambda _: mdl.MultiEncoder(**model_kwargs).to(device).train()),
        list
    )
    model_ids = pipe(
        range(num_ensamble),
        map(lambda _: uuid.uuid4()),
        list
    )

    model_paths = pipe(
        model_ids,
        map(lambda x: model_dir / f'{x}-landsat.pt'),
        list,
    )

    pos_set = pipe(
        range(neg_scale),
        map(lambda _: sets['train_pos']),
        reduce(lambda x, y: x+y)
    )

    train_pos_loader = DataLoader(
        pos_set,
        batch_size=batch_size // 2,
        shuffle=True,
        pin_memory=True,
    )
    train_neg_loaders = pipe(
        range(num_ensamble),
        map(lambda x: DataLoader(
            sets['train_neg'],
            batch_size=batch_size//2,
            pin_memory=True,
            sampler=ChunkSampler(
                epoch_size=len(pos_set),
                len_indices=len(sets['train_neg']),
                shuffle=True,
                start_at=x,
            ),
        )),
        list,
    )
    val_set = sets['val_neg']+sets['val_pos']

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
    )

    batch_len = len(train_pos_loader)
    max_val_score = 0
    for epoch in range(epochs):
        sum_train_loss = 0
        sum_val_loss = 0
        sum_train_score = 0
        sum_val_score = 0
        val_probs = []
        val_labels = []
        train_probs = []
        train_labels = []

        traineds = pipe(
            zip(models, train_neg_loaders),
            map(lambda x: train_epoch(
                model=x[0],
                neg_loader=x[1],
                pos_loader=train_pos_loader,
                landsat_weight=landsat_weight,
                device=device,
                lr=lr
            )),
            list,
        )
        train_metrics = pipe(
            traineds,
            map(lambda x: x[1]),
            reduce(lambda x, y: {
                'landsat': (x['landsat'] + y['landsat'])/2,
                'fusion': (x['fusion'] + y['fusion'])/2,
            }),
        )
        models = pipe(
            traineds,
            map(lambda x: x[0]),
            list,
        )

        val_metrics = validate(
            models=models,
            loader=val_loader,
        )

        with SummaryWriter(log_dir) as w:
            w.add_scalar('train/ssim', train_metrics['ssim'], epoch)
            w.add_scalar('val/ssim', val_metrics['ssim'], epoch)

            if max_val_score <= val_metrics['ssim']:
                max_val_score = val_metrics['ssim']
                pipe(
                    zip(models, model_paths),
                    map(lambda x: torch.save(
                        x[0].landsat_enc.state_dict(), x[1])),
                    list
                )

                pipe(
                    model_ids,
                    map(lambda x: dump_json(model_dir / f'{x}.json', {
                        **val_metrics,
                        "id": str(x),
                    })),
                    list
                )

    return model_dir
