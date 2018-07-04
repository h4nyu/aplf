#!/usr/bin/env python
# -*- coding: utf-8 -*-
from cytoolz.curried import keymap, filter, pipe, merge, map
from dask import delayed
import pandas as pd
from pprint import pprint
from .dataset import TitanicDataset
from .preprocess import label_encode
from .train import train
from .predict import predict


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


base_dir = '/data/titanic'



test_df = delayed(pd.read_csv)('/data/titanic/test.csv')
train_df = delayed(pd.read_csv)('/data/titanic/train.csv')

train_columns = [
    'Pclass',
    'Sex',
    'SibSp',
    'Embarked',
    'Parch',
    'Survived'
]

train_x_columns = [
    'Pclass',
    'Sex',
    'SibSp',
    'Embarked',
    'Parch'
]
train_x_classes = [
    (1, 2, 3),
    ('male', 'female'),
    pipe(range(10),
         tuple),
    ('S', 'C', 'Q'),
    pipe(range(7),
         tuple),
]

train_df = delayed(lambda x: x[train_columns])(train_df)
train_df = delayed(lambda x: x.dropna())(train_df)
train_x_series = pipe(train_x_columns,
                      map(lambda c: delayed(lambda x: x[c])(train_df)),
                      lambda x: zip(x, train_x_classes),
                      map(lambda x: label_encode(*x)),
                      list)


train_y_columns = ['Survived']
train_y_classes = [(0, 1)]
train_y_series = pipe(train_y_columns,
                      map(lambda c: delayed(lambda x: x[c])(train_df)),
                      lambda x: zip(x, train_y_classes),
                      map(lambda x: label_encode(*x)),
                      list)


train_dataset = delayed(TitanicDataset)(
    train_x_series,
    train_x_classes,
    train_y_series,
    train_y_classes
)
train_result = train(train_dataset)
loss_plot = plot(delayed(lambda x: x[1])(train_result), f'{base_dir}/loss.png')
