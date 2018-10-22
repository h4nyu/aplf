from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk, last
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
from . import model as mdl
from aplf.utils import EarlyStop
from aplf import config
from tensorboardX import SummaryWriter
from .metric import iou
from os import path
from .utils import AverageMeter
from .losses import lovasz_softmax, FocalLoss, LossSwitcher, LinearLossSwitcher
from .ramps import linear_rampup
from .preprocess import hflip, add_noise
from aplf.utils import skip_if_exists
from aplf.optimizers import Eve
from .data import ChunkSampler


def get_current_consistency_weight(epoch, weight, rampup):
    return weight * linear_rampup(epoch, rampup)


def update_ema_variables(model, ema_model, alpha):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        return ema_model


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def validate(x, y, epoch):
    score = pipe(
        zip(
            x.argmax(dim=1).cpu().detach().numpy(),
            y.cpu().detach().numpy()
        ),
        map(lambda x: iou(*x)),
        list,
        np.mean
    )
    return score


@skip_if_exists('model_path')
def base_train(model_path,
               sets,
               model_type,
               model_kwargs,
               epochs,
               batch_size,
               log_dir,
               erase_num,
               erase_p,
               consistency_loss_wight,
               center_loss_weight,
               seg_loss_weight,
               ):

    device = torch.device("cuda")
    Model = getattr(mdl, model_type)

    model = Model(**model_kwargs).to(device).train()

    train_pos_loader = DataLoader(
        sets['train_pos'],
        batch_size=batch_size//2,
        shuffle=True,
        pin_memory=True,
    )

    train_neg_loader = DataLoader(
        sets['train_neg'],
        batch_size=batch_size//2,
        pin_memory=True,
        sampler=ChunkSampler(
            epoch_size=len(sets['train_pos']),
            len_indices=len(sets['train_neg']),
            shuffle=True
        ),
    )

    val_loader = DataLoader(
        sets['val_pos'] + sets['val_neg'],
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    class_criterion = nn.CrossEntropyLoss(size_average=True)
    optimizer = optim.Adam(model.parameters(), amsgrad=True)
    train_len = len(train_pos_loader)
    val_len = len(val_loader)

    max_iou_val = 0
    max_iou_train = 0
    min_val_loss = 0

    for epoch in range(epochs):
        sum_class_loss = 0
        sum_train_loss = 0
        sum_train_score = 0
        sum_val_loss = 0
        sum_val_score = 0
        sum_consistency_loss = 0
        sum_seg_loss = 0
        for pos_sample, neg_sample in zip(train_pos_loader, train_neg_loader):
            train_after = torch.cat(
                [pos_sample['after'], neg_sample['after']],
                dim=0
            ).to(device)
            train_before = torch.cat(
                [pos_sample['before'], neg_sample['before']],
                dim=0
            ).to(device)
            train_label = torch.cat(
                [pos_sample['label'], neg_sample['label']],
                dim=0
            ).to(device)

            train_out = model(
                train_before,
                train_after,
            )
            print(train_label)

            class_loss = class_criterion(
                train_out,
                train_label
            )

            loss = class_loss
            sum_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_iou_train = sum_train_score / train_len
        mean_train_loss = sum_train_loss / train_len

        for val_sample in val_loader:
            with torch.no_grad():
                val_before = val_sample['before'].to(device)
                val_after = val_sample['after'].to(device)
                val_lable = val_sample['label'].to(device)

                val_out = model(
                    val_before,
                    val_after
                )
                val_loss = class_criterion(
                    val_out,
                    val_lable
                )
                sum_val_loss += val_loss.item()

        mean_val_loss = sum_val_loss / val_len
        mean_iou_val = sum_val_score / val_len

        with SummaryWriter(log_dir) as w:
            w.add_scalars(
                'iou',
                {
                    'val': mean_iou_val,
                    'train': mean_iou_train,
                },
                epoch
            )
            w.add_scalars(
                'loss',
                {
                    'train': mean_train_loss,
                    'val': mean_val_loss,
                },
                epoch
            )
            w.add_scalar('iou/diff', mean_iou_train - mean_iou_val, epoch)
            w.add_scalar('lr', get_learning_rate(optimizer), epoch)

            if max_iou_val <= mean_iou_val:
                max_iou_val = mean_iou_val
                w.add_text(
                    'iou', f'val: {mean_iou_val}, train: {mean_iou_train}, val_loss:{mean_val_loss}, train_loss:{mean_train_loss}', epoch)
                torch.save(model, model_path)

            if max_iou_train <= mean_iou_train:
                max_iou_train = mean_iou_train
    return model_path
