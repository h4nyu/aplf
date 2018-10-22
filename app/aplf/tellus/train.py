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


class CyclicLR(object):

    def __init__(self,
                 min_factor,
                 max_factor,
                 period,
                 milestones,
                 turning_point,
                 ):
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.period = period
        self.milestones = milestones
        self.turning_point = turning_point
        self.range = self.max_factor - self.min_factor



    def __call__(self, epoch):
        cyclic = 1.0
        phase = epoch % self.period
        turn_phase, ratio = self.turning_point
        turn_cyclic = self.min_factor + self.range * ratio


        if  phase <= turn_phase:
            cyclic = (
                self.min_factor +
                (turn_cyclic - self.min_factor) *
                phase/turn_phase
            )

        else:
            cyclic = turn_cyclic + \
                (self.max_factor - turn_cyclic) * \
                (phase - turn_phase)/(self.period - turn_phase)

        gamma = pipe(
            self.milestones,
            filter(lambda x: x[0] <= epoch),
            map(lambda x: x[1]),
            last
        )
        return cyclic * gamma



@skip_if_exists('model_path')
def base_train(model_path,
               train_set,
               val_set,
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
    print(train_set.__dict__)
    assert False

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    val_batch_size = int(batch_size *
                         len(val_set.indices) / len(train_set.indices))

    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=True,
        pin_memory=True,
    )

    class_criterion = nn.CrossEntropyLoss(size_average=True)
    optimizer = optim.Adam(model.parameters(), amsgrad=True)
    len_batch = min(
        len(train_loader),
        len(val_loader)
    )

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
        for train_sample,  val_sample, in zip(train_loader,  val_loader):
            train_after = train_sample['after'].to(device)
            train_before = train_sample['before'].to(device)
            train_label = train_sample['label'].to(device)
            train_out = model(
                train_before,
                train_after,
            )


            class_loss = class_criterion(
                train_out,
                train_label
            )

            loss = class_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

        mean_iou_val = sum_val_score / len_batch
        mean_iou_train = sum_train_score / len_batch
        mean_train_loss = sum_train_loss / len_batch
        mean_val_loss = sum_val_loss / len_batch

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
                w.add_text('iou', f'val: {mean_iou_val}, train: {mean_iou_train}, val_loss:{mean_val_loss}, train_loss:{mean_train_loss}' , epoch)
                torch.save(model, model_path)

            if max_iou_train <= mean_iou_train:
                max_iou_train = mean_iou_train
    return model_path
