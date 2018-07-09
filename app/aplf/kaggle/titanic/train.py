from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from .model import TitanicNet


def train(model_path, loss_path, dataset):
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    device = torch.device("cuda")
    head_x, head_y = dataset[0]
    model = TitanicNet(input_len=len(head_x)).to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    losses = []
    df = pd.DataFrame(columns=['loss'])
    for e in range(50):
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(output, label)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    torch.save(model, model_path)
    df['loss'] = losses
    df.to_json(loss_path)
    return (model_path, loss_path)
