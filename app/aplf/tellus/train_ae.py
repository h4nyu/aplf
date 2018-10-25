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


def validate(prob, label):
    return iou(
        prob,
        label
    )


@skip_if_exists('model_path')
def train_ae(model_path,
             sets,
             model_type,
             model_kwargs,
             epochs,
             batch_size,
             log_dir,
             rgb_loss_weight,
             pos_loss_weight,
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

    image_criterion = nn.MSELoss(size_average=True)
    optimizer = optim.Adam(model.parameters(), amsgrad=True, lr=0.0001)
    batch_len = len(train_pos_loader)

    max_val_score = 0
    max_iou_train = 0
    min_vial_loss = 0

    for epoch in range(epochs):
        sum_val_score = 0
        sum_train_neg_loss = 0
        sum_train_pos_loss = 0
        sum_val_neg_loss = 0
        sum_val_pos_loss = 0
        sum_train_loss = 0
        train_neg_center_loss = 0
        sum_train_neg_center_loss = 0
        for pos_sample, neg_sample, val_pos_sample, val_neg_sample in zip(train_pos_loader, train_neg_loader, val_pos_loader, val_neg_loader):
            train_palser_neg = torch.cat(
                [
                    neg_sample['palser_before'], neg_sample['palser_after']
                ],
                dim=1
            ).to(device)
            train_landsat_neg = torch.cat(
                [
                    neg_sample['landsat_after'], neg_sample['landsat_before']
                ],
                dim=1
            ).to(device)


            train_palser_neg_out, train_landsat_neg_out = model(
                train_palser_neg,
            )
            train_neg_loss = image_criterion(
                train_palser_neg_out,
                train_palser_neg
            )
            sum_train_neg_loss += train_neg_loss.item()

            train_neg_center_loss = rgb_loss_weight * image_criterion(
                train_landsat_neg_out,
                train_landsat_neg,
            )
            sum_train_neg_center_loss += train_neg_center_loss.item()

            train_palser_pos = torch.cat(
                [
                    pos_sample['palser_before'], pos_sample['palser_after']
                ],
                dim=1
            ).to(device)
            train_palser_pos_out, _ = model(
                train_palser_pos,
            )
            train_pos_loss =  image_criterion(
                train_palser_pos_out,
                train_palser_pos
            )
            sum_train_pos_loss += train_pos_loss.item()

            loss = train_neg_loss + train_neg_center_loss
            if train_pos_loss.item() < train_neg_loss.item():
                print('add pos loss')
                loss += - pos_loss_weight * train_pos_loss

            sum_train_loss += loss.item()

            threshold = 1.1*(train_pos_loss).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                val_palser_neg = torch.cat(
                    [
                        val_neg_sample['palser_before'], val_neg_sample['palser_after']
                    ],
                    dim=1
                ).to(device)

                val_palser_neg_out, _ = model(
                    val_palser_neg,
                )
                val_neg_loss = image_criterion(
                    val_palser_neg_out,
                    val_palser_neg
                )
                sum_val_neg_loss += val_neg_loss.item()

                val_palser_pos = torch.cat(
                    [
                        val_pos_sample['palser_before'], val_pos_sample['palser_after']
                    ],
                    dim=1
                ).to(device)

                val_palser_pos_out, _ = model(
                    val_palser_pos,
                )
                val_pos_loss = image_criterion(
                    val_palser_pos_out,
                    val_palser_pos
                )
                sum_val_pos_loss += val_pos_loss.item()

                val_label = torch.cat(
                    [val_pos_sample['label'], val_neg_sample['label']],
                    dim=0
                ).to(device)
                val_out = pipe(
                    zip(
                        torch.cat(
                            [val_palser_pos_out, val_palser_neg_out], dim=0),
                        torch.cat(
                            [val_palser_pos, val_palser_neg], dim=0),
                    ),
                    map(lambda x: F.mse_loss(*x).item()),
                    list,
                    np.array,
                )

                val_score = validate(
                    val_out > threshold,
                    val_label.cpu().detach().numpy(),
                )
                print(val_out)
                print(f'threshold: {threshold}, score: {val_score}')
                sum_val_score += val_score

        mean_train_neg_loss = sum_train_neg_loss / batch_len
        mean_train_pos_loss = sum_train_pos_loss / batch_len
        mean_train_neg_center_loss = sum_train_neg_center_loss / batch_len
        mean_val_neg_loss = sum_val_neg_loss / batch_len
        mean_val_pos_loss = sum_val_pos_loss / batch_len
        mean_train_loss = sum_train_loss / batch_len
        mean_val_score = sum_val_score / batch_len

        with SummaryWriter(log_dir) as w:
            w.add_scalars(
                'loss/palser',
                {
                    'train-neg': mean_train_neg_loss,
                    'train-pos': mean_train_pos_loss,
                    'val-neg': mean_val_neg_loss,
                    'val-pos': mean_val_pos_loss,
                },
                epoch
            )

            w.add_scalars(
                'score',
                {
                    'val': mean_val_score,
                },
                epoch
            )

            if max_val_score <= mean_val_score:
                max_val_score = mean_val_score
                w.add_text(
                    'iou',
                    f'val: {mean_val_score}, epoch: {epoch}',
                    epoch
                )
                torch.save(model, model_path)

    return model_path
