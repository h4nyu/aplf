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
    return nn.CrossEntropyLoss(size_average=True)(x, y)


def validate(model,
             loader):
    model = model.eval()
    device = torch.device("cuda")
    sum_loss = 0
    epoch_len = len(loader)
    y_preds = []
    y_trues = []
    for sample in loader:
        with torch.no_grad():
            palsar = sample['palsar'].to(device)
            labels = sample['label'].to(device)
            label_preds = model(palsar)
            y_preds += label_preds.argmax(dim=1).cpu().detach().tolist()
            y_trues += labels.cpu().detach().tolist()

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
                lr,
                ):
    model = model.train()
    batch_len = len(pos_loader)

    optimizer = optim.Adam(
        model.fusion_enc.parameters(),
        amsgrad=True,
        lr=lr
    )
    sum_loss = 0
    for pos_sample, neg_sample in zip(pos_loader, neg_loader):
        palsar = torch.cat(
            [pos_sample['palsar'], neg_sample['palsar']],
            dim=0
        ).to(device)
        labels = torch.cat(
            [pos_sample['label'], neg_sample['label']],
            dim=0
        ).to(device)

        loss = criterion(model(palsar), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    mean_loss = sum_loss / batch_len
    return model, {"class": mean_loss}


@skip_if_exists('model_path')
def train(model_path,
          sets,
          model_kwargs,
          epochs,
          batch_size,
          landsat_model_path,
          lr,
          neg_scale,
          log_dir,
          ):
    device = torch.device("cuda")
    model = mdl.MultiEncoder(**model_kwargs).to(device).train()
    model.landsat_enc.load_state_dict(torch.load(landsat_model_path))
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
    train_neg_loader = DataLoader(
        sets['train_neg'],
        batch_size=batch_size//2,
        pin_memory=True,
        sampler=ChunkSampler(
            epoch_size=len(pos_set),
            len_indices=len(sets['train_neg']),
            shuffle=True,
            start_at=0,
        ),
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

        model, train_metrics = train_epoch(
            model=model,
            neg_loader=train_neg_loader,
            pos_loader=train_pos_loader,
            device=device,
            lr=lr
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
        )

        with SummaryWriter(log_dir) as w:
            w.add_scalar('train/class', train_metrics['class'], epoch)
            w.add_scalar('val/iou', val_metrics['iou'], epoch)
            w.add_scalar('val/tpr', val_metrics['tpr'], epoch)
            w.add_scalar('val/fpr', val_metrics['fpr'], epoch)
            w.add_scalar('val/acc', val_metrics['acc'], epoch)

            if max_val_score <= val_metrics['iou']:
                max_val_score = val_metrics['iou']
                torch.save(model.fusion_enc.state_dict(), model_path)
                dump_json(f'{model_path}.json', {**val_metrics})

    return model_path
