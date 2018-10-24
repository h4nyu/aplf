from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk, last
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
from . import model as mdl
from aplf.utils import EarlyStop
from aplf import config
from tensorboardX import SummaryWriter
from .metric import iou
from os import path
from .utils import AverageMeter
from .losses import lovasz_softmax, FocalLoss, LossSwitcher, LinearLossSwitcher, lovasz_softmax_flat
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


def validate(x, y):
    return iou(
        x.argmax(dim=1).cpu().detach().numpy(),
        y.cpu().detach().numpy()
    )


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
               rgb_loss_weight,
               lr,
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

    val_batch_size = batch_size * len(sets['val_pos'])//len(sets['train_pos'])

    val_pos_loader = DataLoader(
        sets['val_pos'],
        batch_size=val_batch_size,
        shuffle=True,
        pin_memory=True,
    )

    val_neg_loader = DataLoader(
        sets['val_neg'],
        batch_size=val_batch_size,
        pin_memory=True,
        sampler=ChunkSampler(
            epoch_size=len(sets['val_pos']),
            len_indices=len(sets['val_neg']),
            shuffle=True
        ),
    )

    class_criterion = lovasz_softmax_flat
    image_criterion = nn.MSELoss(size_average=True)
    optimizer = optim.Adam(model.parameters(), amsgrad=True, lr=0.0001)
    batch_len = len(train_pos_loader)

    max_iou_val = 0
    max_iou_train = 0
    min_vial_loss = 0

    for epoch in range(epochs):
        sum_class_loss = 0
        sum_train_loss = 0
        sum_train_score = 0
        sum_val_loss = 0
        sum_val_score = 0
        sum_consistency_loss = 0
        sum_seg_loss = 0
        for pos_sample, neg_sample, val_pos_sample, val_neg_sample in zip(train_pos_loader, train_neg_loader, val_pos_loader, val_neg_loader):
            train_before = torch.cat(
                [pos_sample['palser_before'], neg_sample['palser_before']],
                dim=0
            ).to(device)
            train_after = torch.cat(
                [pos_sample['palser_after'], neg_sample['palser_after']],
                dim=0
            ).to(device)
            train_label = torch.cat(
                [pos_sample['label'], neg_sample['label']],
                dim=0
            ).to(device)
            train_b_rgb = torch.cat(
                [pos_sample['landsat_before'], neg_sample['landsat_before']],
                dim=0
            ).to(device)
            train_a_rgb = torch.cat(
                [pos_sample['landsat_after'], neg_sample['landsat_after']],
                dim=0
            ).to(device)

            train_out, train_b_rgb_out, train_a_rgb_out = model(
                train_before,
                train_after,
            )
            class_loss = class_criterion(
                train_out,
                train_label
            )

            t_a_rgb_loss = image_criterion(
                train_a_rgb_out,
                train_a_rgb
            )

            t_b_rgb_loss = image_criterion(
                train_b_rgb_out,
                train_b_rgb
            )
            t_rgb_loss = t_a_rgb_loss + t_b_rgb_loss
            loss = class_loss + t_rgb_loss
            sum_train_loss += loss.item()

            train_score = validate(
                train_out,
                train_label,
            )
            sum_train_score += train_score

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                val_before = torch.cat(
                    [val_pos_sample['palser_before'], val_neg_sample['palser_before']],
                    dim=0
                ).to(device)
                val_after = torch.cat(
                    [val_pos_sample['palser_after'], val_neg_sample['palser_after']],
                    dim=0
                ).to(device)
                val_label = torch.cat(
                    [val_pos_sample['label'], val_neg_sample['label']],
                    dim=0
                ).to(device)

                val_out, val_b_rgb_out, val_a_rgb_out = model(
                    val_before,
                    val_after
                )
                val_loss = class_criterion(
                    val_out,
                    val_label
                )

                sum_val_loss += val_loss.item()

                val_score = validate(
                    val_out,
                    val_label,
                )
                sum_val_score += val_score

        mean_iou_train = sum_train_score / batch_len
        mean_train_loss = sum_train_loss / batch_len
        mean_val_loss = sum_val_loss / batch_len
        mean_iou_val = sum_val_score / batch_len

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
                w.add_text('iou', f'val: {mean_iou_val}, train: {mean_iou_train}, val_loss:{mean_val_loss}, train_loss:{mean_train_loss}', epoch)
                torch.save(model, model_path)

            if max_iou_train <= mean_iou_train:
                max_iou_train = mean_iou_train
    return model_path
