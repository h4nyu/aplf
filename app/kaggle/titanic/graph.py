#!/usr/bin/env python
# -*- coding: utf-8 -*-
from distributed import Client
from torch.utils.data import Dataset, DataLoader
from dask import delayed
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
def preprocess(df):
    df['Age'] = df['Age'] / 100
    df['Pclass'] = df['Pclass'] / 3
    df['Fare'] = df['Fare'] / 1000
    return df


class TitanicDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df, is_train=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx][['Age', 'SibSp',
                                  'Fare', 'Pclass']].values.astype(float)
        if self.is_train:
            label = self.df.iloc[idx][['Survived']]
            return torch.FloatTensor(data), torch.FloatTensor([float(label)])
        else:
            return torch.FloatTensor(data)


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


if __name__ == '__main__':
    test_df = delayed(pd.read_csv)('/data/titanic/test.csv')
    train_df = delayed(pd.read_csv)('/data/titanic/train.csv')
    train_dataset = delayed(TitanicDataset)(preprocess(train_df))
    test_dataset = delayed(TitanicDataset)(preprocess(test_df), is_train=False)
    train_result = train(train_dataset)
    loss_plot = plot(delayed(lambda x: x[1])(train_result), '/data/loss.png')
    predict_result = predict(
        delayed(lambda x: x[0])(train_result), test_dataset)

    with Client('dask-scheduler:8786') as c:
        predict_result.visualize('/data/titanic/graph.svg')
        result = c.compute(predict_result, sync=True)
