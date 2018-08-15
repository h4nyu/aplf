import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from .model import TgsSaltcNet
from aplf.utils import EarlyStop


def train(model_path,
          train_dataset,
          val_dataset,
          epochs,
          patience,
          batch_size,
          ):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_batch_size = int(batch_size * len(val_dataset) / len(train_dataset))
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=True
    )
    model = TgsSaltcNet().to(device)
    model.train()

    optimizer = optim.Adam(model.parameters())
    critertion = nn.NLLLoss(
        size_average=True
    )
    losses = []
    val_losses = []
    df = pd.DataFrame()
    el = EarlyStop(patience, base_size=5)
    for e in range(epochs):
        for (train_id, train_depth, train_image, train_mask), (val_id, val_depth, val_image, val_mask) in zip(train_loader, val_loader):
            train_image = train_image.to(device)
            val_image = val_image.to(device)
            train_mask = train_mask.to(device).view(-1, 101, 101).long()
            val_mask = val_mask.to(device).view(-1, 101, 101).long()

            optimizer.zero_grad()
            output = model(train_image)
            loss = critertion(
                output,
                train_mask
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print(loss.item())

            output = model(val_image)
            val_loss = critertion(
                output,
                val_mask
            )
            val_losses.append(val_loss.item())
            is_overfit = el(val_loss.item())
            if is_overfit:
                break
        if is_overfit:
            break

    torch.save(model, model_path)

    df['train'] = losses
    df['val'] = val_losses
    return {
        'model_path': model_path,
        'progress': df
    }
