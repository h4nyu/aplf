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
from ..losses import lovasz_softmax, FocalLoss, LossSwitcher, LinearLossSwitcher, lovasz_softmax_flat

from ..preprocess import batch_aug
from aplf.utils import skip_if_exists
from aplf.optimizers import Eve
from ..data import ChunkSampler, Augment


def validate(predicts, loader):

    y_preds = np.array(predicts).mean(axis=0).argmax(axis=1)
    y_trues = pipe(
        loader,
        map(lambda x: x['label'].cpu().detach().tolist()),
        reduce(lambda x, y: x+y),
        np.array,
    )

    score = iou(
        y_preds,
        y_trues,
    )
    tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()
    return {
        'TPR': tp/(tp+fn),
        'FNR': fn/(tp+fn),
        'FPR': fp/(fp+tn),
        'acc': (tp+tn) / (tp+tn+fp+fn),
        'pre': tp / (tp + fp),
        'iou': tp / (fn+tp+fp),
    }


def validate_epoch(model,
            loader,
            ):
    y_preds = []
    device = torch.device('cuda')
    for sample in loader:
        with torch.no_grad():
            palsar_x = sample['palsar'].to(device)
            label_preds = model(palsar_x)[0].softmax(dim=1)
            y_preds += label_preds.cpu().detach().tolist()

    return y_preds


def train_epoch(model,
                criterion,
                pos_loader,
                neg_loader,
                lr,
                ):

    device = torch.device("cuda")
    optimizer = optim.Adam(model.parameters(), amsgrad=True, lr=lr)
    batch_len = len(pos_loader)
    sum_train_loss = 0
    aug = batch_aug(Augment())
    for pos_sample, neg_sample in zip(pos_loader, neg_loader):
        palsar_x = torch.cat(
            [pos_sample['palsar'], neg_sample['palsar']],
            dim=0
        )
        palsar_x = aug(palsar_x, ch=1).to(device)
        landsat_x = torch.cat(
            [pos_sample['landsat'], neg_sample['landsat']],
            dim=0
        )
        landsat_x = aug(landsat_x, ch=3).to(device)
        labels = torch.cat(
            [pos_sample['label'], neg_sample['label']],
            dim=0
        ).to(device)

        logit_loss = criterion(
            model(palsar_x),
            (labels, landsat_x),
        )

        loss = logit_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_train_loss += loss.item()
    mean_loss = sum_train_loss / batch_len
    return model, mean_loss


@curry
def criterion(landsat_weight, x, y):
    logit, landsat_x = x
    labels, landsat_y = y
    image_cri = nn.MSELoss(size_average=True)
    class_cri = nn.CrossEntropyLoss(size_average=True)
    loss = class_cri(logit, labels) + landsat_weight * \
        image_cri(landsat_x, landsat_y)
    return loss


@skip_if_exists('model_dir')
def train_multi(model_dir,
                sets,
                model_type,
                model_kwargs,
                epochs,
                batch_size,
                log_dir,
                val_batch_size,
                landsat_weight,
                lr,
                num_ensamble,
                divides,
                ):

    model_dir = Path(model_dir)
    model_dir.mkdir()

    Model = getattr(mdl, model_type)
    device = torch.device("cuda")
    models = pipe(
        range(num_ensamble),
        map(lambda _: Model(**model_kwargs).to(device).train()),
        list
    )

    model_paths = pipe(
        range(num_ensamble),
        map(lambda x: model_dir / f'{x}.pt'),
        list,
    )

    pipe(
        zip(models, model_paths),
        map(lambda x: torch.save(*x)),
        list
    )

    pos_set = pipe(
        range(divides),
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
            sampler=ChunkSampler(
                epoch_size=len(pos_set),
                len_indices=len(sets['train_neg']),
                shuffle=True,
                start_at=x,
            ),
            pin_memory=True,
        )),
        list,
    )

    val_loader = DataLoader(
        sets['val_neg']+sets['val_pos'],
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=True,
    )
    batch_len = len(train_pos_loader)

    max_val_score = 0
    max_iou_train = 0
    min_vial_loss = 0
    mean_train_pos_loss = 0
    mean_train_neg_loss = 0
    mean_val_pos_loss = 0
    mean_val_neg_loss = 0
    min_train_pos_loss = 1
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
            map(lambda x: delayed(train_epoch)(
                model=x[0],
                neg_loader=x[1],
                pos_loader=train_pos_loader,
                criterion=criterion(landsat_weight),
                lr=lr
            )),
            list,
        )

        train_loss = pipe(
            traineds,
            map(delayed(lambda x: x[1])),
            list,
            delayed(np.mean)
        )

        models = pipe(
            traineds,
            map(delayed(lambda x: x[0])),
            list,
        )
        metrics = pipe(
            models,
            map(lambda x: delayed(validate_epoch)(
                model=x,
                loader=val_loader,
            )),
            list,
            lambda x: delayed(validate)(
                predicts=x,
                loader=val_loader
            )
        )

        models, train_loss, metrics = dask.compute(models, train_loss, metrics)

        with SummaryWriter(log_dir) as w:
            w.add_scalars(
                'loss',
                {
                    'train': train_loss,
                },
                epoch
            )
            iou = metrics['iou']
            w.add_scalars(
                'score',
                {
                    **metrics
                },
                epoch
            )

            if max_val_score <= iou:
                max_val_score = iou
                w.add_text(
                    'iou',
                    f'val: {iou}, epoch: {epoch}',
                    epoch
                )
                pipe(
                    zip(models, model_paths),
                    map(lambda x: torch.save(x[0], x[1])),
                    list
                )
    return model_dir
