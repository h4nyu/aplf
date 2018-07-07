from torch.utils.data import DataLoader
from dask import delayed
import torch
import torch.optim as optim
import torch.nn as nn
from .model import TitanicNet


@delayed
def train(dataset):
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    device = torch.device("cuda")
    head_x, head_y = dataset[0]
    print(len(head_x))
    model = TitanicNet(input_len=len(head_x)).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    losses = []
    for e in range(100):
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if (batch_idx % 100 == 0):
                print(loss.item())
    return (model, losses)
