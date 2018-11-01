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
             ):
    device = torch.device("cuda")
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
                reduce(lambda x, y: (x+y)/2),
                lambda x: x.argmax(dim=1),
            )
            y_preds += label_preds.cpu().detach().tolist()
            y_trues += labels.cpu().detach().tolist()
            batch_len += 1

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


def train_epoch(model_path,
                criterion,
                pos_loader,
                neg_loader,
                landsat_weight,
                lr,
                ):

    device = torch.device("cuda")
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
<<<<<<< HEAD
            (labels, landsat_x),
            landsat_weight,
||||||| merged common ancestors
            (labels, landsat_x)
=======
            (labels, landsat_x),
>>>>>>> f00fca2c78f359ef94ef9b5673e5456fc35500a8
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


@curry
def criterion(landsat_weight, x, y):
    image_cri = nn.MSELoss(size_average=True)
    class_cri = nn.CrossEntropyLoss(size_average=True)
    logit, landsat_x = x
    labels, landsat_y = y
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
<<<<<<< HEAD
                val_batch_size,
                landsat_weight,
||||||| merged common ancestors
                landsat_weight,
=======
>>>>>>> f00fca2c78f359ef94ef9b5673e5456fc35500a8
                lr,
                num_ensamble,
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
    check_model_paths = pipe(
        range(num_ensamble),
        map(lambda x: model_dir / f'{x}_check.pt'),
        list,
    )

    pipe(
        zip(models, model_paths),
        map(lambda x: torch.save(*x)),
        list
    )

    pos_set = pipe(
        range(150 // (num_ensamble + 1)),
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
        batch_size=val_batch_size,
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
            map(lambda x: train_epoch(
                model_path=x[0],
                neg_loader=x[1],
                pos_loader=train_pos_loader,
<<<<<<< HEAD
                landsat_weight=landsat_weight,
                criterion=criterion,
||||||| merged common ancestors
                criterion=criterion,
                device=device,
=======
                criterion=criterion(landsat_weight),
                device=device,
>>>>>>> f00fca2c78f359ef94ef9b5673e5456fc35500a8
                lr=lr
            )),
            list,
        )

        train_loss = pipe(
            traineds,
            map(lambda x: x[1]),
            list,
            np.mean
        )

        metrics = validate(
            model_paths=model_paths,
            loader=val_loader,
<<<<<<< HEAD
            criterion=criterion,
||||||| merged common ancestors
            criterion=criterion,
            device=device
=======
            device=device
>>>>>>> f00fca2c78f359ef94ef9b5673e5456fc35500a8
        )

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
                    model_paths,
                    map(torch.load),
                    lambda x: zip(x, check_model_paths),
                    map(lambda x: torch.save(x[0], x[1])),
                    list
                )
    return model_dir
