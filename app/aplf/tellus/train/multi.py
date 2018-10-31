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
from aplf.utils import skip_if_exists
from aplf.optimizers import Eve
from ..data import ChunkSampler


def validate(model_paths,
             loader,
             criterion,
             device):
    models = pipe(
        model_paths,
        map(torch.load),
        map(lambda x: x.eval().to(device)),
        list
    )
    y_preds = []
    y_trues = []
    sum_loss = 0
    batch_len = 0
    for sample in loader:
        with torch.no_grad():
            palsar_x = sample['palsar'].to(device)
            landsat_y = sample['landsat'].to(device)
            labels = sample['label'].to(device)
            label_preds = pipe(
                models,
                map(lambda x: x(palsar_x)[0].softmax(dim=1)),
                reduce(lambda x, y: torch.max(x, y))
            )
            y_preds += label_preds.argmax(dim=1).cpu().detach().tolist()
            y_trues += labels.cpu().detach().tolist()
            batch_len += 1

    score = iou(
        y_preds,
        y_trues,
    )

    return confusion_matrix(y_trues, y_preds).ravel()


def train_epoch(model_path,
                criterion,
                pos_loader,
                neg_loader,
                device,
                lr,
                ):
    model = torch.load(model_path)
    optimizer = optim.Adam(model.parameters(), amsgrad=True, lr=lr)
    batch_len = len(pos_loader)
    sum_train_loss = 0
    for pos_sample, neg_sample in zip(pos_loader, neg_loader):
        palsar_x = torch.cat(
            [pos_sample['palsar'], neg_sample['palsar']],
            dim=0
        ).to(device)
        landsat_x = torch.cat(
            [pos_sample['landsat'], neg_sample['landsat']],
            dim=0
        ).to(device)
        labels = torch.cat(
            [pos_sample['label'], neg_sample['label']],
            dim=0
        ).to(device)

        logit_loss = criterion(
            model(palsar_x),
            (labels, landsat_x)
        )

        loss = logit_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_train_loss += loss.item()
    mean_loss = sum_train_loss / batch_len
    torch.save(model, model_path)
    return model_path, mean_loss


def aug(x):
    pass


def criterion(x, y):
    image_cri = nn.MSELoss(size_average=True)
    class_cri = nn.CrossEntropyLoss(size_average=True)
    logit, landsat_x = x
    labels, landsat_y = y
    loss = class_cri(logit, labels) + 0.5*image_cri(landsat_x, landsat_y)
    return loss


@skip_if_exists('model_dir')
def train_multi(model_dir,
                sets,
                model_type,
                model_kwargs,
                epochs,
                batch_size,
                log_dir,
                rgb_loss_weight,
                lr,
                num_ensamble=2,
                ):

    model_dir = Path(model_dir)
    model_dir.mkdir()
    device = torch.device("cuda")
    Model = getattr(mdl, model_type)

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
        range(150//(num_ensamble + 1)),
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

    val_loader = DataLoader(
        sets['val_neg']+sets['val_pos'],
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
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
            zip(model_paths, train_neg_loaders),
            map(lambda x: delayed(train_epoch)(
                model_path=x[0],
                neg_loader=x[1],
                pos_loader=train_pos_loader,
                criterion=criterion,
                device=device,
                lr=lr
            )),
            list,
            lambda x: dask.compute(*x)
        )

        train_loss = pipe(
            traineds,
            map(lambda x: x[1]),
            list,
            np.mean
        )

        tn, fp, fn, tp = validate(
            model_paths=model_paths,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        with SummaryWriter(log_dir) as w:
            w.add_scalars(
                'loss',
                {
                    'train': train_loss,
                },
                epoch
            )
            iou = tp / (fn+tp+fp)
            w.add_scalars(
                'score',
                {
                    'TPR': tp/(tp+fn),
                    'FNR': fn/(tp+fn),
                    'FPR': fp/(fp+tn),
                    'acc': (tp+tn) / (tp+tn+fp+fn),
                    'pre': tp / (tp + fp),
                    'iou': tp / (fn+tp+fp),
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

    return model_path
