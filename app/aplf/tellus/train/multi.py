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
from ..data import ChunkSampler, Augment, batch_aug
import uuid


def validate(models,
             loader,
             ):
    models = pipe(
        models,
        map(lambda x: x.eval()),
        list
    )

    device = torch.device("cuda")
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
                reduce(lambda x, y: (x+y)/2)
            )
            y_preds += label_preds.argmax(dim=1).cpu().detach().tolist()
            y_trues += labels.cpu().detach().tolist()
            batch_len += 1

    score = iou(
        y_preds,
        y_trues,
    )

    tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()
    return {
        'tpr': tp/(tp+fn),
        'fnp': fn/(tp+fn),
        'fpr': fp/(fp+tn),
        'acc': (tp+tn) / (tp+tn+fp+fn),
        'pre': tp / (tp + fp),
        'iou': tp / (fn+tp+fp),
    }


def train_epoch(model,
                pos_loader,
                neg_loader,
                device,
                lr
                ):
    model = model.train()
    batch_len = len(pos_loader)
    landstat_optim = optim.Adam(
        model.landsat_enc.parameters(),
        amsgrad=True,
        lr=lr
    )
    fusion_optim = optim.Adam(
        model.fusion_enc.parameters(),
        amsgrad=True,
        lr=lr
    )

    image_cri = nn.MSELoss(size_average=True)
    class_cri = nn.CrossEntropyLoss(size_average=True)

    sum_fusion_loss = 0
    sum_landsat_loss = 0
    for pos_sample, neg_sample in zip(pos_loader, neg_loader):

        aug = Augment()
        palsar_x = torch.cat(
            [pos_sample['palsar'], neg_sample['palsar']],
            dim=0
        )
        palsar_x = batch_aug(aug, palsar_x, ch=1).to(device)
        landsat_x = torch.cat(
            [pos_sample['landsat'], neg_sample['landsat']],
            dim=0
        )
        landsat_x = batch_aug(aug, landsat_x, ch=3).to(device)
        labels = torch.cat(
            [pos_sample['label'], neg_sample['label']],
            dim=0
        ).to(device)

        landsat_loss = image_cri(model(palsar_x)[1], landsat_x)
        landstat_optim.zero_grad()
        landsat_loss.backward()
        landstat_optim.step()

        fusion_loss = class_cri(model(palsar_x)[0], labels)
        fusion_optim.zero_grad()
        fusion_loss.backward()
        fusion_optim.step()

        sum_fusion_loss += fusion_loss.item()
        sum_landsat_loss += landsat_loss.item()
    mean_fusion_loss = sum_fusion_loss / batch_len
    mean_landsat_loss = sum_landsat_loss / batch_len
    return model, {"fusion": mean_fusion_loss, "landsat": mean_landsat_loss}


@skip_if_exists('model_dir')
def train_multi(model_dir,
                sets,
                model_type,
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
        map(lambda x: model_dir / f'{uuid.uuid4()}.pt'),
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
            map(lambda x: train_epoch(
                model=x[0],
                neg_loader=x[1],
                pos_loader=train_pos_loader,
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
            w.add_scalar('loss/fusion', train_metrics['fusion'], epoch)
            w.add_scalar('loss/landsat', train_metrics['landsat'], epoch)
            w.add_scalar('val/iou', val_metrics['iou'], epoch)
            w.add_scalar('val/tpr', val_metrics['tpr'], epoch)
            w.add_scalar('val/fpr', val_metrics['fpr'], epoch)
            w.add_scalar('val/acc', val_metrics['acc'], epoch)

            if max_val_score <= val_metrics['iou']:
                max_val_score = val_metrics['iou']
                w.add_text(
                    'iou',
                    f"val: {val_metrics['iou']}, epoch: {epoch}",
                    epoch
                )
                pipe(
                    zip(models, model_paths),
                    map(lambda x: torch.save(x[0], x[1])),
                    list
                )

    return model_dir
