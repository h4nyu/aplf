from torch.utils.data import DataLoader
from dask import delayed
import torch
import torch.optim as optim
import torch.nn as nn
from .model import TitanicNet


@delayed
def train(dataset):
    loader = DataLoader(dataset, batch_size=1)
    device = torch.device("cpu")
    model = TitanicNet(input_len=dataset.x_len,
                       output_len=dataset.y_len).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    losses = []
    for e in range(10):
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return (model, losses)
