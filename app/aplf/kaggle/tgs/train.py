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


def train(model_id,
          model_path,
          train_dataset,
          val_dataset,
          epochs,
          batch_size,
          feature_size,
          patience,
          base_size,
          log_dir,
          ):
    writer = SummaryWriter(config["TENSORBORAD_LOG_DIR"])
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda")

    train_loader = DataLoader(
        train_dataset.train(),
        batch_size=batch_size,
        shuffle=True
    )

    val_batch_size = int(batch_size * len(val_dataset) / len(train_dataset))
    val_loader = DataLoader(
        val_dataset.train(),
        batch_size=val_batch_size,
        shuffle=True
    )
    model = UNet(
        feature_size=feature_size
    ).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters())
    critertion = nn.CrossEntropyLoss()
    el = EarlyStop(patience, base_size=base_size)
    n_itr = 0
    is_overfit = False
    len_batch = len(train_loader)
    max_val_score = 0
    for epoch in range(epochs):
        sum_train_loss = 0
        sum_train_score = 0
        sum_val_loss = 0
        sum_val_score = 0
        batch_idx = 0

        for train_sample, val_sample in zip(train_loader, val_loader):
            train_image = train_sample['image'].to(device)
            val_image = val_sample['image'].to(device)
            train_mask = train_sample['mask'].to(
                device).view(-1, 101, 101).long()
            val_mask = val_sample['mask'].to(device).view(-1, 101, 101).long()

            optimizer.zero_grad()
            output = model(train_image)

            loss = critertion(
                output,
                train_mask
            )
            loss.backward()
            optimizer.step()

            sum_train_loss += loss.item()
            sum_train_score += pipe(
                zip(
                    output.argmax(dim=1).cpu().detach().numpy(),
                    train_sample['mask'].numpy()
                ),
                map(lambda x: iou(*x)),
                list,
                np.mean
            )

            output = model(val_image)
            val_loss = critertion(
                output,
                val_mask
            )
            sum_val_loss += val_loss.item()
            sum_val_score += pipe(
                zip(
                    output.argmax(dim=1).cpu().detach().numpy(),
                    val_sample['mask'].numpy()
                ),
                map(lambda x: iou(*x)),
                list,
                np.mean
            )

        writer.add_scalar(
            f'{log_dir}/train_iou_{model_id}',
            sum_train_score/len_batch,
            epoch
        )
        writer.add_scalar(
            f'{log_dir}/val_iou_{model_id}',
            sum_val_score/len_batch,
            epoch
        )

        writer.add_scalar(
            f'{log_dir}/val_{model_id}',
            sum_val_loss/len_batch,
            epoch
        )
        writer.add_scalar(
            f'{log_dir}/train_{model_id}',
            sum_train_loss/len_batch,
            epoch
        )

        if max_val_score < sum_val_score/len_batch:
            torch.save(model, model_path)
            max_val_score = sum_val_score/len_batch

        if sum_val_score/len_batch > 0:
            is_overfit = el(- sum_val_score/len_batch)
        if is_overfit:
            break

    writer.close()

    return model_path
