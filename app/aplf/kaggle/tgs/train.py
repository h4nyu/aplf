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
               seg_set,
               val_set,
               no_lable_set,
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

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )


    seg_batch_size = int(batch_size *
                         len(seg_set.indices) / len(train_set.indices))
    seg_loader = DataLoader(
        seg_set,
        batch_size=seg_batch_size,
        shuffle=True,
    )

    val_batch_size = int(batch_size *
                         len(val_set.indices) / len(train_set.indices))

    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=True
    )
    no_label_loader = DataLoader(
        no_lable_set,
        batch_size=val_batch_size,
        shuffle=True
    )

    class_criterion = nn.CrossEntropyLoss(size_average=True)
    seg_criterion = lovasz_softmax
    consistency_criterion = nn.MSELoss(size_average=True)
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
        sum_center_loss = 0
        sum_train_score = 0
        sum_val_loss = 0
        sum_val_score = 0
        sum_consistency_loss = 0
        sum_seg_loss = 0
        for train_sample, seg_sample, val_sample, no_label_sample in zip(train_loader, seg_loader, val_loader, no_label_loader):
            train_image = train_sample['image'].to(device)
            train_mask = train_sample['mask'].to(device)
            train_out, train_center_out = model(
                add_noise(
                    train_image,
                    erase_num=erase_num,
                    erase_p=erase_p,
                )
            )

            train_score = validate(
                train_out,
                train_mask.view(-1, 101, 101).long(),
                epoch,
            )

            class_loss = seg_criterion(
                train_out,
                train_mask.view(-1, 101, 101).long()
            )

            center_loss = center_loss_weight * class_criterion(
                train_center_out,
                F.max_pool2d(train_mask, kernel_size=101).view(-1, 1, 1).long()
            )

            with torch.no_grad():
                val_image = val_sample['image'].to(device)
                val_mask = val_sample['mask'].to(device)
                no_label_image = no_label_sample['image'].to(device)
                consistency_input = torch.cat([
                    train_image[0:val_batch_size],
                    val_image,
                    no_label_image,
                ], dim=0)

                tea_out, tea_center_out = model(
                    consistency_input.flip([3]),
                )

            stu_out, stu_center_out = model(
                add_noise(
                    consistency_input,
                    erase_num=erase_num,
                    erase_p=erase_p,
                )
            )
            consistency_loss = consistency_loss_wight * (
                consistency_criterion(
                    stu_out.softmax(dim=1),
                    tea_out.flip([3]).softmax(dim=1)
                ) +
                consistency_criterion(
                    stu_center_out.softmax(dim=1),
                    tea_center_out.softmax(dim=1)
                )
            )

            seg_image = seg_sample['image'].to(device)
            seg_out, _  = model(
                add_noise(
                    seg_image,
                    erase_num=erase_num,
                    erase_p=erase_p,
                )
            )
            seg_mask = seg_sample['mask'].to(device)
            seg_loss = seg_loss_weight * seg_criterion(
                seg_out,
                seg_mask.view(-1, *seg_out.size()[2:]).long()
            )


            loss = class_loss + consistency_loss + seg_loss + center_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            sum_train_score += train_score
            sum_class_loss += class_loss.item()
            sum_consistency_loss += consistency_loss.item()
            sum_center_loss += center_loss.item()
            sum_train_loss += loss.item()
            sum_seg_loss += seg_loss.item()

            with torch.no_grad():
                val_out, _ = model(val_image)
                val_loss = class_criterion(
                    val_out,
                    val_mask.view(-1, *val_out.size()[2:]).long()
                )
                val_score = validate(
                    val_out,
                    val_mask.view(-1, *val_out.size()[2:]).long(),
                    epoch,
                )
                sum_val_loss += val_loss.item()
                sum_val_score += val_score

        mean_iou_val = sum_val_score / len_batch
        mean_iou_train = sum_train_score / len_batch
        mean_train_loss = sum_train_loss / len_batch
        mean_val_loss = sum_val_loss / len_batch
        mean_class_loss = sum_class_loss / len_batch
        mean_center_loss = sum_center_loss / len_batch
        mean_consistency_loss = sum_consistency_loss / len_batch
        mean_seg_loss = sum_seg_loss / len_batch

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
            w.add_scalar('loss/consistency', mean_consistency_loss, epoch)
            w.add_scalar('loss/class', mean_class_loss, epoch)
            w.add_scalar('loss/center', mean_center_loss, epoch)
            w.add_scalar('loss/seg', mean_seg_loss, epoch)
            w.add_scalar('loss/diff', mean_val_loss - mean_class_loss, epoch)


            if max_iou_val >= mean_iou_val:
                max_iou_val = mean_iou_val
                w.add_text('iou', f'val: {mean_iou_val}, train: {mean_iou_train}, val_loss:{mean_val_loss}, train_loss:{mean_train_loss}' , epoch)
                torch.save(model, model_path)

            if max_iou_train <= mean_iou_train:
                max_iou_train = mean_iou_train
    return model_path
