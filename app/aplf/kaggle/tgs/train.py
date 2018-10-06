from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk, last
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
                 ):
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.period = period
        self.milestones = milestones

    def __call__(self, epoch):
        cyclic = self.min_factor + \
            (self.max_factor - self.min_factor) * \
            (epoch % self.period)/self.period
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
               ):
    device = torch.device("cuda")
    Model = getattr(mdl, model_type)

    model = Model(**model_kwargs).to(device).train()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )

    val_batch_size = int(batch_size *
                         len(val_set.indices) / len(train_set.indices))

    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=True
    )
    class_criterion = lovasz_softmax
    optimizer = optim.Adam(model.parameters())
    len_batch = min(
        len(train_loader),
        len(val_loader)
    )

    max_iou_val = 0
    max_iou_train = 0

    for epoch in range(epochs):
        sum_class_loss = 0
        sum_train_loss = 0
        sum_center_loss = 0
        sum_train_score = 0
        sum_val_loss = 0
        sum_val_score = 0
        for train_sample, val_sample in zip(train_loader, val_loader):
            train_image = train_sample['image'].to(device)
            train_mask = train_sample['mask'].to(device)
            val_image = val_sample['image'].to(device)
            val_mask = val_sample['mask'].to(device)
            train_out, train_center_out = model(
                add_noise(
                    train_image,
                    num=erase_num,
                )
            )

            train_score = validate(
                train_out,
                train_mask.view(-1, *train_out.size()[2:]).long(),
                epoch,
            )

            class_loss = class_criterion(
                train_out,
                train_mask.view(-1, *train_out.size()[2:]).long()
            )

            train_center_mask = F.interpolate(train_mask, size=train_center_out.size()[2:])
            center_loss = class_criterion(
                train_center_out,
                train_center_mask.view(-1, *train_center_out.size()[2:]).long(),
            )
            loss = class_loss + center_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            sum_train_score += train_score
            sum_class_loss += class_loss.item()
            sum_center_loss += center_loss.item()
            sum_train_loss += loss.item()

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

        with SummaryWriter(log_dir) as w:
            w.add_scalars(
                'iou',
                {
                    'val': mean_iou_val,
                    'train': mean_iou_train,
                    'diff': mean_iou_train - mean_iou_val,
                },
                epoch
            )
            w.add_scalars(
                'loss',
                {
                    'train': mean_train_loss,
                    'class': mean_class_loss,
                    'center': mean_center_loss,
                    'val': mean_val_loss,
                    'diff': mean_val_loss - mean_class_loss,
                },
                epoch
            )
            w.add_scalars(
                'lr',
                {
                    'lr': get_learning_rate(optimizer),
                },
                epoch
            )

            if max_iou_val <= mean_iou_val:
                max_iou_val = mean_iou_val
                w.add_text('iou', f'val: {mean_iou_val}, train: {mean_iou_train}:' , epoch)
                torch.save(model, model_path)

            if max_iou_train <= mean_iou_train:
                max_iou_train = mean_iou_train
    return model_path

@skip_if_exists('out_model_path')
def fine_train(in_model_path,
               out_model_path,
               train_set,
               val_set,
               no_labeled_set,
               epochs,
               labeled_batch_size,
               no_labeled_batch_size,
               log_dir,
               consistency,
               cyclic_period,
               milestones,
               erase_num,
               ):
    device = torch.device("cuda")
    model = torch.load(in_model_path).to(device).train()

    train_loader = DataLoader(
        train_set,
        batch_size=labeled_batch_size,
        shuffle=True,
    )

    no_labeled_dataloader = DataLoader(
        no_labeled_set,
        batch_size=no_labeled_batch_size,
        shuffle=True
    )

    val_batch_size = int(labeled_batch_size *
                         len(val_set.indices) / len(train_set.indices))

    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=True
    )

    class_criterion = lovasz_softmax
    consistency_criterion = nn.MSELoss(size_average=True)

    optimizer = optim.Adam(model.parameters())
    len_batch = min(
        len(train_loader),
        len(no_labeled_dataloader),
        len(val_loader)
    )
    scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=CyclicLR(
            min_factor=0.2,
            max_factor=1,
            period=cyclic_period,
            milestones=milestones,
        )
    )

    max_iou_val = 0
    max_iou_train = 0

    for epoch in range(epochs):
        sum_class_loss = 0
        sum_train_loss = 0
        sum_consistency_loss = 0
        sum_center_loss = 0
        sum_train_score = 0
        sum_val_loss = 0
        sum_val_score = 0
        for train_sample, no_labeled_sample, val_sample in zip(train_loader, no_labeled_dataloader, val_loader):
            train_image = train_sample['image'].to(device)
            train_mask = train_sample['mask'].to(device)
            val_image = val_sample['image'].to(device)
            val_mask = val_sample['mask'].to(device)
            no_labeled_image = no_labeled_sample['image'].to(device)

            train_image = add_noise(
                train_image,
                num=erase_num
            )
            train_out, train_center_out = model(train_image)

            train_score = validate(
                train_out,
                train_mask.view(-1, *train_out.size()[2:]).long(),
                epoch,
            )

            class_loss = class_criterion(
                train_out,
                train_mask.view(-1, *train_out.size()[2:]).long(),
            )

            train_center_mask = F.interpolate(train_mask, size=train_center_out.size()[2:])
            center_loss = class_criterion(
                train_center_out,
                train_center_mask.view(-1, *train_center_out.size()[2:]).long(),
            )
            with torch.no_grad():
                consistency_input = torch.cat([
                    val_image,
                    no_labeled_image,
                ])
                tea_out, _ = model(
                    add_noise(
                        consistency_input,
                        num=erase_num
                    )
                )

            stu_out, _ = model(
                add_noise(
                    consistency_input,
                    num=erase_num
                )
            )

            consistency_loss = consistency * \
                consistency_criterion(
                    stu_out.softmax(dim=1),
                    tea_out.softmax(dim=1),
                )

            loss = class_loss + center_loss + consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            sum_train_score += train_score
            sum_class_loss += class_loss.item()
            sum_consistency_loss += consistency_loss.item()
            sum_center_loss += center_loss.item()
            sum_train_loss += loss.item()

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

        #  update LR
        #  scheduler.step()

        mean_iou_val = sum_val_score / len_batch
        mean_iou_train = sum_train_score / len_batch
        mean_train_loss = sum_train_loss / len_batch
        mean_val_loss = sum_val_loss / len_batch
        mean_class_loss = sum_class_loss / len_batch
        mean_center_loss = sum_center_loss / len_batch
        mean_consistency_loss = sum_consistency_loss / len_batch

        with SummaryWriter(log_dir) as w:
            w.add_scalars(
                'iou',
                {
                    'val': mean_iou_val,
                    'train': mean_iou_train,
                    'diff': mean_iou_train - mean_iou_val,
                },
                epoch
            )
            w.add_scalars(
                'loss',
                {
                    'train': mean_train_loss,
                    'class': mean_class_loss,
                    'center': mean_center_loss,
                    'consistency': mean_consistency_loss,
                    'val': mean_val_loss,
                    'diff': mean_val_loss - mean_class_loss,
                },
                epoch
            )
            w.add_scalars(
                'lr',
                {
                    'lr': get_learning_rate(optimizer),
                },
                epoch
            )

            if max_iou_val <= mean_iou_val:
                max_iou_val = mean_iou_val
                w.add_text('iou', f'val: {mean_iou_val}, train: {mean_iou_train}:' , epoch)
                torch.save(model, out_model_path)

            if max_iou_train <= mean_iou_train:
                max_iou_train = mean_iou_train
    return out_model_path
