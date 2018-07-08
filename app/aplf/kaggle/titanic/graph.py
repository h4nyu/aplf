#!/ usr / bin / env python
# -*- coding: utf-8 -*-
from cytoolz.curried import keymap, filter, pipe, merge, map, compose
from dask import delayed
import pandas as pd
import numpy as np
from pprint import pprint
from .dataset import TitanicDataset
from .preprocess import label_encode, max_min_scaler, one_hot
from .train import train
from .predict import predict, evaluate


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

test_columns = [
    'Pclass',
    'Sex',
    'SibSp',
    'Embarked',
    'Parch',
    'Age',
    'Fare'
]

train_columns = [
    *test_columns,
    'Survived',
]


x_columns = [
    'Pclass',
    'Sex',
    'SibSp',
    'Embarked',
    'Parch',
]
x_classes = [
    (1, 2, 3),
    ('male', 'female'),
    pipe(range(10),
         tuple),
    ('S', 'C', 'Q'),
    pipe(range(7),
         tuple),
]

get_len = compose(list, map(len))
train_df = delayed(lambda x: x[train_columns])(train_df)
train_df = delayed(lambda x: x.dropna())(train_df)

test_df = delayed(lambda x: x[test_columns])(test_df)
test_df = delayed(lambda x: x.dropna())(test_df)

train_x = pipe(x_columns,
               map(lambda c: delayed(lambda x: x[c])(train_df)),
               lambda x: zip(x, x_classes),
               map(lambda x: label_encode(*x)),
               lambda x: zip(x, get_len(x_classes)),
               map(lambda x: one_hot(*x)),
               list)


train_x += pipe(['Age', 'Fare'],
                map(lambda c: delayed(lambda x: x[c])(train_df)),
                map(max_min_scaler),
                list)


train_y = delayed(lambda x: x['Survived'].values)(train_df)
train_y = delayed(one_hot)(train_y, 2)

train_dataset = delayed(TitanicDataset)(
    train_x,
    train_y,
)

test_x = pipe(x_columns,
              map(lambda c: delayed(lambda x: x[c])(test_df)),
              lambda x: zip(x, x_classes),
              map(lambda x: label_encode(*x)),
              lambda x: zip(x, get_len(x_classes)),
              map(lambda x: one_hot(*x)),
              list)
test_x += pipe(['Age', 'Fare'],
               map(lambda c: delayed(lambda x: x[c])(test_df)),
               map(max_min_scaler),
               list)

test_dataset = delayed(TitanicDataset)(test_x)


train_result = train(train_dataset)
loss_plot = plot(delayed(lambda x: x[1])(train_result), f'{base_dir}/loss.png')

predict_result = delayed(predict)(
    delayed(lambda x: x[0])(train_result),
    test_dataset
)

eval_result = delayed(evaluate)(
    delayed(lambda x: x[0])(predict_result),
    delayed(lambda x: x[1])(predict_result),
)
