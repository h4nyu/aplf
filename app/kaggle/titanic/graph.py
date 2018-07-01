#!/usr/bin/env python
# -*- coding: utf-8 -*-
from distributed import Client
from torch.utils.data import Dataset
from dask import delayed
import pandas as pd


@delayed
def preprocess(df):
    df['Age'] = df['Age'] / 100
    df['Pclass'] = df['Pclass'] / 3
    df['Fare'] = df['Fare'] / 1000
    return df


class TitanicDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx][['Age', 'SibSp', 'Fare', 'Pclass']].values
        label = self.df.iloc[idx]['Survived']
        return data, label


if __name__ == '__main__':
    test_csv = delayed(pd.read_csv)('/data/titanic/test.csv')
    train_df = delayed(pd.read_csv)('/data/titanic/train.csv')
    preprocessed = preprocess(train_df)
    train_dataset = delayed(TitanicDataset)(preprocessed)

    with Client('dask-scheduler:8786') as c:
        train_dataset.visualize('/data/graph.svg')

        result = c.compute(train_dataset, sync=True)
        print(result[0])
