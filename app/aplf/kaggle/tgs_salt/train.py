import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device("cuda")


def train(model_path,
          train_dataset,
          val_dataset,
          batch_size=8):

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

    for e in range(30):
        for x_train, x_val in zip(train_loader, val_loader):
            return x_train
    #  sample_x, sample_y = dataset[0]
    #  model = TitanicNet(
    #      input_len=sample_x.shape[0],
    #  ).to(device)
    #  model.train()
    #  optimizer = optim.Adam(model.parameters())
    #  losses = []
    #  df = pd.DataFrame(columns=['loss'])
    #  critertion = nn.NLLLoss()
    #          data, label = data.to(device), label.to(device)
    #          optimizer.zero_grad()
    #          output = model(data)
    #          loss = critertion(output, label)
    #          loss.backward()
    #          optimizer.step()
    #          losses.append(loss.item())
    #  torch.save(model, model_path)
    #  df['loss'] = losses
    #  df.to_json(loss_path)
    #  return (model_path, loss_path)
