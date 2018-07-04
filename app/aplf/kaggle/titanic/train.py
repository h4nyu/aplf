from torch.utils.data import DataLoader
from dask import delayed
import torch
import torch.optim as optim
import torch.nn as nn
from .model import TitanicNet


@delayed
def train(dataset):
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(len(dataset))
    device = torch.device("cpu")
    head_x, head_y = dataset[0]
    model = TitanicNet(input_len=len(head_x),
                       output_len=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    losses = []
    for e in range(100):
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                losses.append(loss.item())
    return (model, losses)
