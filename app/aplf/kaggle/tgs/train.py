import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from .model import UNet
from aplf.utils import EarlyStop
from aplf import config
from tensorboardX import SummaryWriter


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
    critertion = nn.NLLLoss(
        weight=torch.tensor([1.0, 1.0]).to(device),
        size_average=True,
    )
    el = EarlyStop(patience, base_size=base_size)
    n_itr = 0
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
                F.log_softmax(output, dim=1),
                train_mask
            )
            loss.backward()
            optimizer.step()

            output = model(val_image)
            val_loss = critertion(
                F.log_softmax(output, dim=1),
                val_mask
            )
            is_overfit = el(val_loss.item())
            writer.add_scalar(f'loss/val_{model_id}', val_loss.item(), n_itr)
            writer.add_scalar(f'loss/train_{model_id}', loss.item(), n_itr)
            n_itr += 1

            if is_overfit:
                break
        if is_overfit:
            break
    writer.close()
    torch.save(model, model_path)

    return model_path
