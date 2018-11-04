from pathlib import Path
import dask
from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk, last
from sklearn.metrics import confusion_matrix
import random
import torchvision.utils as vutils
from pathlib import Path
from cytoolz import curry
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
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
from ..data import ChunkSampler, batch_aug, Augment


def validate(models,
             loader,
             device):
    y_preds = []
    y_trues = []
    sum_loss = 0
    batch_len = 0
    for sample in loader:
        with torch.no_grad():
            palsar_x = sample['palsar'].to(device)
            labels = sample['label'].to(device)
            label_preds = pipe(
                models,
                map(lambda x: x(palsar_x)[0].softmax(dim=1)),
                reduce(lambda x, y: (x+y)/2)
            )
            print(label_preds.argmax(dim=1).cpu().detach().tolist())
            print(labels.cpu().detach().tolist())
            y_preds += label_preds.argmax(dim=1).cpu().detach().tolist()
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


def train_epoch(model,
                criterion,
                pos_loader,
                neg_loader,
                optimizer,
                device,
                lr,
                ):
    batch_len = len(pos_loader)
    sum_train_loss = 0
    for pos_sample, neg_sample in zip(pos_loader, neg_loader):
        palsar_x = torch.cat(
            [
                pos_sample['palsar'],
                neg_sample['palsar']
            ],
            dim=0
        ).to(device)
        landsat_x = torch.cat(
            [
                pos_sample['landsat'],
                neg_sample['landsat']
            ],
            dim=0
        ).to(device)

        labels = torch.cat(
            [pos_sample['label'], neg_sample['label']],
            dim=0
        ).to(device)


        label_pred, landsat_pred = model(palsar_x)
        loss = criterion(
            (label_pred, landsat_pred),
            (labels, landsat_x)
        )
        print('-----------train------------')
        print(label_pred.argmax(dim=1).cpu().detach().tolist())
        print(labels.cpu().detach().tolist())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_train_loss += loss.item()
    mean_loss = sum_train_loss / batch_len
    return model, mean_loss


class Criterion(object):
    def __init__(self, landsat_weight):
        self.landsat_weight = landsat_weight
        self.image_cri = nn.MSELoss(size_average=True)
        self.class_cri = nn.CrossEntropyLoss(size_average=True)

    def __call__(self, x, y):
        logit, landsat_x = x
        labels, landsat_y = y
        loss = self.class_cri(logit, labels) + self.landsat_weight * \
            self.image_cri(landsat_x, landsat_y)
        return loss


@skip_if_exists('model_dir')
def train_multi(model_dir,
                sets,
                model_type,
                model_kwargs,
                epochs,
                batch_size,
                val_batch_size,
                log_dir,
                landsat_weight,
                validate_interval,
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
    optimizers = pipe(
        models,
        map(lambda x: optim.Adam(x.parameters(), amsgrad=True, lr=lr)),
        list
    )

    model_paths = pipe(
        range(num_ensamble),
        map(lambda x: model_dir / f'{x}.pt'),
        list,
    )

    pos_set = pipe(
        range(validate_interval),
        map(lambda _: sets['train_pos']),
        reduce(lambda x, y: x+y)
    )
    train_pos_loader = DataLoader(
        pos_set,
        batch_size=batch_size // 2,
        shuffle=True,
        pin_memory=True,
    )
    train_neg_loader = DataLoader(
        sets['train_neg'],
        batch_size=batch_size//2,
        pin_memory=True,
        sampler=RandomSampler(
            data_source=sets['train_neg'],
        ),
    )

    val_loader = DataLoader(
        sets['val_neg']+sets['val_pos'],
        batch_size=val_batch_size,
        pin_memory=True,
        shuffle=False,
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
    criterion = Criterion(landsat_weight)
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
            zip(models, optimizers),
            map(lambda x: train_epoch(
                model=x[0],
                optimizer=x[1],
                neg_loader=train_neg_loader,
                pos_loader=train_pos_loader,
                criterion=criterion,
                device=device,
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

        models = pipe(
            traineds,
            map(lambda x: x[0]),
            list,
        )

        metrics = validate(
            models=models,
            loader=val_loader,
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
            w.add_scalars(
                'score',
                {
                    **metrics
                },
                epoch
            )

            if max_val_score <= metrics['iou']:
                max_val_score = metrics['iou']
                w.add_text(
                    'iou',
                    f"val: {metrics['iou']}, epoch: {epoch}",
                    epoch
                )

    return model_path
