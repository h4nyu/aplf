import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from .model import TgsSaltcNet


def train(model_path,
          train_dataset,
          val_dataset,
          batch_size=16,
          epochs=1):
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
    critertion = nn.MSELoss()
    losses = []
    val_losses = []
    df = pd.DataFrame()
    for e in range(epochs):
        for (train_depth, train_image, train_mask), (val_depth, val_image, val_mask) in zip(train_loader, val_loader):
            train_image, train_mask = train_image.to(device), train_mask.to(device)
            val_image, val_mask = val_image.to(device), val_mask.to(device)
            optimizer.zero_grad()
            output = model(train_image)
            loss = critertion(output, train_mask)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            output = model(val_image)
            val_loss = critertion(output, val_mask)
            val_losses.append(val_loss.item())
    torch.save(model, model_path)

    df['train'] = losses
    df['val'] = val_losses
    return df
