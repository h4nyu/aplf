from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from .model import UNet
from aplf.utils import EarlyStop
from aplf import config
from tensorboardX import SummaryWriter
from .metric import iou
from os import path
from .utils import AverageMeter
from .losses import softmax_mse_loss, lovasz_softmax
from .ramps import sigmoid_rampup
from .preprocess import hflip



def get_current_consistency_weight(epoch, weight, rampup):
    return weight * sigmoid_rampup(epoch, rampup)


def update_ema_variables(model, ema_model, alpha):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        return ema_model

def batch_hflip(images):
    with torch.no_grad():
        return pipe(
            images,
            map(hflip),
            list
        )

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']





def validate(critertion, x, y):
    loss = critertion(
        x,
        y
    )
    score = pipe(
        zip(
            x.argmax(dim=1).cpu().detach().numpy(),
            y.cpu().detach().numpy()
        ),
        map(lambda x: iou(*x)),
        list,
        np.mean
    )
    return loss, score


def train(model_path,
          train_dataset,
          val_dataset,
          no_labeled_dataset,
          epochs,
          labeled_batch_size,
          no_labeled_batch_size,
          feature_size,
          patience,
          reduce_lr_patience,
          base_size,
          log_dir,
          ema_decay,
          consistency,
          consistency_rampup,
          depth,
          ):
    device = torch.device("cuda")

    model = UNet(
        feature_size=feature_size,
        depth=depth,
    ).to(device)
    model.train()
    ema_model = UNet(
        feature_size=feature_size,
        depth=depth,
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.train()

    train_loader = DataLoader(
        train_dataset,
        batch_size=labeled_batch_size,
        shuffle=True
    )

    no_labeled_dataloader = DataLoader(
        no_labeled_dataset,
        batch_size=no_labeled_batch_size,
        shuffle=True
    )

    val_batch_size = int(labeled_batch_size *
                         len(val_dataset) / len(train_dataset))
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=True
    )

    class_criterion = nn.CrossEntropyLoss(size_average=True)
    consistency_criterion = softmax_mse_loss
    optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum=0.9)
    reduce_LR = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'min',
        factor=0.5,
        patience=reduce_lr_patience,
    )
    len_batch = min(
        len(train_loader),
        len(no_labeled_dataloader),
        len(val_loader)
    )

    el = EarlyStop(patience, base_size=base_size)
    max_iou_val = 0
    max_iou_train = 0

    for epoch in range(epochs):
        sum_class_loss = 0
        sum_consistency_loss = 0
        sum_train_loss = 0
        sum_train_score = 0
        sum_val_loss = 0
        sum_val_score = 0
        for train_sample, no_labeled_sample, val_sample in zip(train_loader, no_labeled_dataloader, val_loader):
            train_image = train_sample['image'].to(device)
            train_mask = train_sample['mask'].to(
                device).view(-1, 101, 101).long()
            val_image = val_sample['image'].to(device)
            val_mask = val_sample['mask'].to(device).view(-1, 101, 101).long()
            no_labeled_image = no_labeled_sample['image'].to(device)
            model_out = model(train_image)
            class_loss, train_score = validate(
                class_criterion,
                model_out,
                train_mask
            )
            sum_train_score += train_score

            with torch.no_grad():
                consistency_input = torch.cat([
                    train_image,
                    val_image,
                    no_labeled_image
                ])

                # add hflop noise
                ema_model_out = ema_model(
                    consistency_input.flip([3])
                ).flip([3])

            model_out = model(consistency_input)

            consistency_weight = get_current_consistency_weight(
                epoch=epoch,
                weight=consistency,
                rampup=consistency_rampup,
            )

            consistency_loss = consistency_weight * \
                consistency_criterion(model_out, ema_model_out)
            loss = consistency_loss + class_loss
            sum_class_loss += class_loss.item()
            sum_consistency_loss += consistency_loss.item()
            sum_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                val_loss, val_score = validate(
                    class_criterion,
                    model(val_image),
                    val_mask
                )
                sum_val_loss += val_loss.item()
                sum_val_score += val_score



        print(f"epoch: {epoch} score : {sum_val_score / len_batch}")
        mean_iou_val = sum_val_score / len_batch
        mean_iou_train = sum_train_score / len_batch
        mean_train_loss = sum_train_loss / len_batch
        mean_consistency_loss = sum_consistency_loss / len_batch
        mean_val_loss = sum_val_loss / len_batch
        mean_class_loss = sum_class_loss / len_batch

        reduce_LR.step(mean_val_loss)
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
                    'consistency': mean_consistency_loss,
                    'class': mean_class_loss,
                    'val': mean_val_loss,
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
            torch.save(ema_model, model_path)
            with torch.no_grad():
                ema_model = update_ema_variables(model, ema_model, ema_decay)

        if max_iou_train <= mean_iou_train:
            max_iou_train = mean_iou_train

        if sum_val_score / len_batch > 0:
            is_overfit = el(- mean_iou_val)

        if is_overfit:
            break
    return model_path
