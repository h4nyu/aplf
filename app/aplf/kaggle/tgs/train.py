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
from .losses import softmax_mse_loss


def update_ema_variables(model, ema_model, alpha):
    # Use the true average until the exponential average is more correct
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    return ema_model


def validate(model, critertion, x, y):
    output = model(x)
    loss = critertion(
        output,
        y
    )
    score = pipe(
        zip(
            output.argmax(dim=1).cpu().detach().numpy(),
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
          unsupervised_dataset,
          epochs,
          batch_size,
          feature_size,
          patience,
          base_size,
          log_dir,
          ):
    device = torch.device("cuda")

    model = UNet(
        feature_size=feature_size
    ).to(device)
    model.train()
    ema_model = UNet(
        feature_size=feature_size
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.train()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size // 2,
        shuffle=True
    )

    unsupervised_loader = DataLoader(
        unsupervised_dataset,
        batch_size=batch_size // 2,
        shuffle=True
    )

    val_batch_size = int(batch_size * len(val_dataset) / len(train_dataset))
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=True
    )

    val_batch_size = int(batch_size * len(val_dataset) / len(train_dataset))

    class_criterion = nn.CrossEntropyLoss(size_average=True)
    consistency_criterion = softmax_mse_loss
    optimizer = optim.Adam(model.parameters())
    len_batch = min(
        len(train_loader),
        len(unsupervised_loader),
        len(val_loader)
    )

    el = EarlyStop(patience, base_size=base_size)
    max_val_score = 0

    for epoch in range(epochs):
        sum_class_loss = 0
        sum_consistency_loss = 0
        sum_train_loss = 0
        sum_val_loss = 0
        sum_val_score = 0
        for train_sample, unsupervised_sample, val_sample in zip(train_loader, unsupervised_loader, val_loader):

            train_image = train_sample['image'].to(device)
            train_mask = train_sample['mask'].to(
                device).view(-1, 101, 101).long()
            val_image = val_sample['image'].to(device)
            val_mask = val_sample['mask'].to(device).view(-1, 101, 101).long()
            unsupervised_image = unsupervised_sample['image'].to(device)
            ema_model_out = ema_model(train_image)
            model_out = model(train_image)
            class_loss = class_criterion(model_out, train_mask)
            sum_class_loss += class_loss.item()
            consistency_loss = consistency_criterion(model_out, ema_model_out)
            sum_consistency_loss += consistency_loss.item()
            loss = class_loss + consistency_loss

            ema_model_out = ema_model(unsupervised_image)
            model_out = model(unsupervised_image)
            consistency_loss = consistency_criterion(model_out, ema_model_out)
            loss += consistency_loss
            sum_train_loss += loss.item()
            sum_consistency_loss += consistency_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema_model = update_ema_variables(model, ema_model, 0.8)

            val_loss, val_score = validate(
                model,
                class_criterion,
                val_image,
                val_mask
            )
            print(f'score:{val_score}')
            sum_val_loss += val_loss.item()
            sum_val_score += val_score
        print(f"epoch: {epoch} score : {sum_val_score / len_batch}")

        with SummaryWriter(log_dir) as w:
            w.add_scalars(
                'iou',
                {
                    'val': sum_val_score / len_batch,
                },
                epoch
            )
            w.add_scalars(
                'loss',
                {
                    'train': sum_train_loss / len_batch,
                    'consistency': sum_consistency_loss / len_batch,
                    'class': sum_class_loss / len_batch,
                    'val': sum_val_loss / len_batch,
                },
                epoch
            )

        if max_val_score < sum_val_score / len_batch:
            torch.save(model, model_path)
            max_val_score = sum_val_score / len_batch

        if sum_val_score / len_batch > 0:
            is_overfit = el(- sum_val_score / len_batch)

        if is_overfit:
            break
    return model_path
