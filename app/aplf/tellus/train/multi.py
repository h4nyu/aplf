from pathlib import Path
import dask
from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk, last
from datetime import datetime
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
from ..models import MultiEncoder
from aplf import config
from aplf.preprocess import RandomErasing
from tensorboardX import SummaryWriter
from ..metric import iou
from os import path
from ..losses import lovasz_softmax, FocalLoss, LossSwitcher, LinearLossSwitcher, lovasz_softmax_flat
from aplf.utils import skip_if_exists
from aplf.optimizers import Eve
from ..data import ChunkSampler, Augment, batch_aug
import uuid
import json


def dump_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)
        return path


def validate(model,
             loader,
             ):

    image_cri = nn.MSELoss(size_average=True)
    with torch.no_grad():
        model.eval()
        device = torch.device("cuda")
        y_preds = []
        y_trues = []
        sum_loss = 0
        batch_len = 0
        sum_loss = 0
        for sample in loader:
            palsar = sample['palsar'].to(device)
            labels = sample['label'].to(device)
            landsat = sample['landsat'].to(device)
            label_preds = model(palsar)
            loss = image_cri(model(palsar, part='landsat'), landsat)
            y_preds += label_preds.argmax(dim=1).cpu().detach().tolist()
            y_trues += labels.cpu().detach().tolist()
            sum_loss += loss.item()
            batch_len += 1

        score = iou(
            y_preds,
            y_trues,
        )
        mean_loss = sum_loss / batch_len

        tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()
        return {
            'landsat': mean_loss,
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
                pi_loader,
                device,
                lr
                ):
    aug = RandomErasing(p=1)
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
    sum_pi_loss = 0
    for pos_sample, neg_sample, pi_sample in zip(pos_loader, neg_loader, pi_loader):

        palsar = torch.cat(
            [pos_sample['palsar'], neg_sample['palsar']],
            dim=0
        ).to(device)
        palsar = batch_aug(aug, palsar, ch=1).to(device)
        landsat = torch.cat(
            [pos_sample['landsat'], neg_sample['landsat']],
            dim=0
        ).to(device)
        labels = torch.cat(
            [pos_sample['label'], neg_sample['label']],
            dim=0
        ).to(device)
        pi_palsar = pi_sample['palsar']
        pi_palsar0 = batch_aug(aug, pi_palsar, ch=1).to(device)
        pi_palsar1 = batch_aug(aug, pi_palsar, ch=1).to(device)

        landsat_loss = image_cri(model(palsar, part='landsat'), landsat)
        sum_landsat_loss += landsat_loss.item()
        pi_loss = image_cri(
            model(pi_palsar0, part='landsat'),
            model(pi_palsar1, part='landsat')
        )
        sum_pi_loss += pi_loss.item()
        loss = landsat_loss + 0.1 * pi_loss
        landstat_optim.zero_grad()
        loss.backward()
        landstat_optim.step()

        fusion_loss = class_cri(model(palsar), labels)
        sum_fusion_loss += fusion_loss.item()
        fusion_optim.zero_grad()
        fusion_loss.backward()
        fusion_optim.step()

    mean_fusion_loss = sum_fusion_loss / batch_len
    mean_landsat_loss = sum_landsat_loss / batch_len
    mean_pi_loss = sum_pi_loss / batch_len
    return model, {"fusion": mean_fusion_loss, "landsat": mean_landsat_loss, "pi": mean_pi_loss}


@skip_if_exists('model_path')
def train_multi(model_path,
                sets,
                model_kwargs,
                epochs,
                batch_size,
                log_dir,
                landsat_weight,
                lr,
                neg_scale,
                ):

    model_path = Path(model_path)

    device = torch.device("cuda")
    model = MultiEncoder(**model_kwargs).to(device).train()
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
        sampler=ChunkSampler(epoch_size=len(pos_set),
                             len_indices=len(sets['train_neg']),
                             shuffle=True,
                             ),
    )
    val_set = sets['val_neg']+sets['val_pos']

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
    )

    pi_loader = DataLoader(
        val_set,
        batch_size=len(val_loader) * batch_size // len(train_pos_loader),
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

        model, train_metrics = train_epoch(
            model=model,
            neg_loader=train_neg_loader,
            pos_loader=train_pos_loader,
            pi_loader=pi_loader,
            device=device,
            lr=lr
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
        )

        with SummaryWriter(log_dir) as w:
            w.add_scalar('train/fusion', train_metrics['fusion'], epoch)
            w.add_scalar('train/landsat', train_metrics['landsat'], epoch)
            w.add_scalar('train/pi', train_metrics['pi'], epoch)
            w.add_scalar('val/iou', val_metrics['iou'], epoch)
            w.add_scalar('val/tpr', val_metrics['tpr'], epoch)
            w.add_scalar('val/fpr', val_metrics['fpr'], epoch)
            w.add_scalar('val/acc', val_metrics['acc'], epoch)
            w.add_scalar('val/landsat', val_metrics['landsat'], epoch)

            if max_val_score <= val_metrics['iou']:
                max_val_score = val_metrics['iou']
                w.add_text(
                    'iou',
                    f"val: {val_metrics['iou']}, epoch: {epoch}",
                    epoch
                )
                torch.save(model, model_path)
                dump_json(f'{model_path}.json', {
                    **val_metrics,
                    "create_date": datetime.now().isoformat(),
                })

    return model_path
