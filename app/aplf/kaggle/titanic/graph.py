#!/usr/bin/env python
# -*- coding: utf-8 -*-
from cytoolz.curried import keymap, filter, pipe, merge, map
from torch.utils.data import Dataset, DataLoader
from dask import delayed
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pprint import pprint
from sklearn import preprocessing
from .dataset import TitanicDataset


@delayed
def predict(model, dataset):
    loader = DataLoader(dataset)
    device = torch.device("cpu")
    labels = []
    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        output = model(data)
        labels.append(output.detach().numpy()[0][0])
    return labels


@delayed
def label_encode(series, classes):
    le = preprocessing.LabelEncoder()
    le.fit(classes)
    return le.transform(series)


class TitanicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(4, 4)
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


@delayed
def plot(x,
         path,
         **kwargs):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x)
    plt.savefig(path)
    plt.close()
    return path


@delayed
def train(dataset):
    loader = DataLoader(dataset)
    device = torch.device("cpu")
    model = TitanicNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    losses = []
    for e in range(3):
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(output, label)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return (model, losses)


test_df = delayed(pd.read_csv)('/data/titanic/test.csv')
train_df = delayed(pd.read_csv)('/data/titanic/train.csv')

columns = ['Pclass', 'Sex']
classes = [(1, 2, 3), ('male', 'female')]
series = pipe(columns,
              map(lambda c: delayed(lambda x: x[c])(train_df)),
              lambda x: zip(x, classes),
              map(lambda x: label_encode(*x)),
              list)

train_dataset = delayed(TitanicDataset)(series, classes,)
test_dataset = delayed(TitanicDataset)(test_df, is_train=False)
train_result = train(train_dataset)
loss_plot = plot(delayed(lambda x: x[1])(train_result), '/data/loss.png')
predict_result = predict(delayed(lambda x: x[0])(train_result), test_dataset)
