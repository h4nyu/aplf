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
          patience,
          base_size,
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
    model = UNet().to(device)
    model.train()

    optimizer = optim.Adam(model.parameters())
    critertion = nn.CrossEntropyLoss()
    el = EarlyStop(patience, base_size=base_size)
    n_itr = 0
    is_overfit = False
    for e in range(epochs):
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

            score = pipe(
                zip(
                    output.argmax(dim=1).cpu().detach().numpy(),
                    train_sample['mask'].numpy()
                ),
                map(lambda x: iou(*x)),
                list,
                np.mean
            )
            writer.add_scalar(f'loss/train_iou_{model_id}',
                               score,
                               n_itr
                              )
            print(f"train_iou: {score}")
            output = model(val_image)
            val_loss = critertion(
                output,
                val_mask
            )
            writer.add_scalar(f'loss/val_{model_id}', val_loss.item(), n_itr)
            writer.add_scalar(f'loss/train_{model_id}', loss.item(), n_itr)
            score = pipe(
                zip(
                    output.argmax(dim=1).cpu().detach().numpy(),
                    val_sample['mask'].numpy()
                ),
                map(lambda x: iou(*x)),
                list,
                np.mean
            )
            if score > 0:
                is_overfit = el(-score)
            print(f"val_iou: {score}")
            writer.add_scalar(
                f'loss/val_iou_{model_id}',
                score,
                n_itr
            )
            n_itr += 1

            if is_overfit:
                break
        if is_overfit:
            break
    writer.close()
    torch.save(model, model_path)

    return model_path
