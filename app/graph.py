from distributed import Client, progress
from joblib import Memory
from dask import visualize
from time import sleep
from cytoolz.curried import pipe, map, take
import torch.nn.functional as F
import dask.bag as db
import numpy as np
import random
import dask.array as da
import torch
import torch.optim as optim
from pprint import pprint
from aplf.utils import dict_kwargs
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from torch.utils.data import DataLoader
from aplf.dataset import DummyDataset
from aplf.model import Autoencoder

memory = Memory(cachedir='/', verbose=0)


@dict_kwargs
def gen_dataset(annomalies,
                start,
                stop,
                window_size,
                num,
                chunks):
    return DummyDataset(start=start,
                        stop=stop,
                        num=num,
                        chunks=chunks,
                        window_size=window_size,
                        annomalies=annomalies,
                        transform=None)


@dict_kwargs
def model_train(dataset,
                window_size,
                lr):
    device = torch.device("cuda")

    model = Autoencoder(window_size).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    losses = []

    for epoch in range(0, 1):
        for batch_idx, (data, label) in enumerate(loader):
            input_data, target = data.to(device), data.to(device)
            optimizer.zero_grad()
            output = model(input_data)
            loss = F.l1_loss(output,
                             target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(dataset),
                    100. * batch_idx / len(loader), loss.item()))
    return model


@dict_kwargs
def plot_dataset(data,
                 from_idx,
                 to_idx,
                 path):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    y = data[from_idx:to_idx]
    plt.plot(y)
    plt.savefig(path)
    return path


window_size = 10

dsk = {
    "gen_dataset": (gen_dataset, {
        'annomalies': [(0, 1000), (2000, 4000)],
        'start':  0,
        'stop':  100,
        'num':  100000,
        'window_size':  window_size,
        'chunks':  2,
    }),

    "model_train": (model_train, "gen_dataset", {
        "lr": 0.01,
        'window_size': window_size
    }),
    "plot_dataset": (plot_dataset, "gen_dataset", {
        'from_idx': 0,
        "to_idx": 10000,
        'path': '/data/plot.png'
    }),
}


with Client('dask_scheduler:8786') as c:
    try:
        result = c.get(dsk, 'model_train')

    finally:
        c.restart()
