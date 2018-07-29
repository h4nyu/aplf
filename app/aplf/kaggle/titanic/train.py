import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from .model import TitanicNet


def train(model_path, loss_path, dataset):
    loader = DataLoader(dataset, batch_size=3, shuffle=True)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda")

    sample_x, sample_y = dataset[0]
    model = TitanicNet(
        input_len=sample_x.shape[0],
    ).to(device)
    model.train()
    optimizer = optim.Adam(model.parameters())
    losses = []
    df = pd.DataFrame(columns=['loss'])
    critertion =  nn.NLLLoss()
    for e in range(20):
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = critertion(output, label)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    torch.save(model, model_path)
    df['loss'] = losses
    df.to_json(loss_path)
    return (model_path, loss_path)
