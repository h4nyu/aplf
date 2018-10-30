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
from ..models import FusionNet
from aplf.utils import EarlyStop
from aplf import config
from tensorboardX import SummaryWriter
from ..metric import iou
from os import path
from ..losses import lovasz_softmax, FocalLoss, LossSwitcher, LinearLossSwitcher, lovasz_softmax_flat
from aplf.utils import skip_if_exists
from aplf.optimizers import Eve
from ..data import ChunkSampler


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


def extract_sample(pos_sample, neg_sample):
    with torch.no_grad():
        palser_pos_x = torch.cat(
            [pos_sample['palser_before'], pos_sample['palser_after']],
            dim=1,
        )

        palser_neg_x = torch.cat(
            [neg_sample['palser_before'], neg_sample['palser_after']],
            dim=1,
        )
        palser_x = torch.cat(
            [palser_pos_x, palser_neg_x],
            dim=0,
        )
        labels = torch.cat(
            [pos_sample['label'], neg_sample['label']],
            dim=0
        )

        landsat_pos_x = torch.cat(
            [pos_sample['landsat_before'], pos_sample['landsat_after']],
            dim=1
        )

        landsat_neg_x = torch.cat(
            [neg_sample['landsat_before'], neg_sample['landsat_after']],
            dim=1
        )
        landsat_x = torch.cat(
            [landsat_pos_x, landsat_neg_x],
            dim=0,
        )
        return palser_x, landsat_x, labels


@skip_if_exists('model_path')
def train(model_path,
          sets,
          model_type,
          model_kwargs,
          epochs,
          batch_size,
          log_dir,
          lr,
          ):

    device = torch.device("cuda")
    model = FusionNet(**model_kwargs).to(device).train()

    train_pos_loader = DataLoader(
        sets['train_pos'],
        batch_size=batch_size // 2,
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
        batch_size=batch_size // 50,
        shuffle=True,
        pin_memory=True,
    )

    val_neg_loader = DataLoader(
        sets['val_neg'],
        batch_size=batch_size,
        pin_memory=True,
        sampler=ChunkSampler(
            epoch_size=len(sets['val_pos']),
            len_indices=len(sets['val_neg']),
            shuffle=True
        ),
    )

    class_criterion = lovasz_softmax_flat
    optimizer = optim.Adam(model.parameters(), amsgrad=True, lr=lr)
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
        batch_len = 0
        val_probs = []
        val_labels = []
        train_probs = []
        train_labels = []
        for pos_sample, neg_sample, val_pos_sample, val_neg_sample in zip(train_pos_loader, train_neg_loader, val_pos_loader, val_neg_loader):
            start = random.randint(0, batch_size//2 - 1)
            end = random.randint(batch_size//2, batch_size - 1)
            palser_x, landsat_x, labels = pipe(
                extract_sample(pos_sample, neg_sample),
                map(lambda x: x.to(device)),
                tuple
            )
            logit_out = model(
                palser_x,
                landsat_x
            )
            logit_loss = class_criterion(
                logit_out,
                labels
            )
            loss = logit_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_train_loss += loss.item()
            train_score = validate(
                logit_out.argmax(dim=1).cpu().detach().tolist(),
                labels.cpu().detach().tolist()
            )
            sum_train_score += train_score

            with torch.no_grad():
                palser_x, landsat_x, labels = pipe(
                    extract_sample(val_pos_sample, val_neg_sample),
                    map(lambda x: x.to(device)),
                    tuple
                )
                logit_out = model(
                    palser_x,
                    landsat_x,
                )

                val_loss = class_criterion(
                    logit_out,
                    labels
                )
                sum_val_loss += val_loss.item()

                val_probs = logit_out.argmax(dim=1).cpu().detach().tolist()
                val_labels = labels.cpu().detach().tolist()
                val_score = validate(
                    val_probs,
                    val_labels,
                )
                sum_val_score += val_score
                print(val_probs)
                print(val_labels)
                print(val_score)
                batch_len += 1

        mean_train_score = sum_train_score / batch_len
        mean_train_loss = sum_train_loss / batch_len
        mean_val_loss = sum_val_loss / batch_len
        mean_iou_val = sum_val_score / batch_len
        mean_val_score = sum_val_score / batch_len

        with SummaryWriter(log_dir) as w:
            w.add_scalars(
                'loss',
                {
                    'train': mean_train_loss,
                    'val': mean_val_loss,
                },
                epoch
            )

            w.add_scalars(
                'score',
                {
                    'val': mean_val_score,
                    'train': mean_train_score,
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

            if min_train_pos_loss >= mean_train_pos_loss:
                min_train_pos_loss = mean_train_pos_loss

    return model_path
