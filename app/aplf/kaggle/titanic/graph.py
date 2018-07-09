#!/ usr / bin / env python
# -*- coding: utf-8 -*-
from cytoolz.curried import keymap, filter, pipe, merge, map, compose
from dask import delayed
import pandas as pd
from pprint import pprint
from .dataset import TitanicDataset

from .preprocess import (
    label_encode,
    max_min_scaler,
    one_hot,
    name_to_class,
    string_len,
    cabin_to_class,
    string_int,
    smooth_outer,
    ticket_to_class
)
from .train import train
from .predict import predict, evaluate


base_dir = '/store/titanic'


preprocess_params = {
    'Fsize': {
        'funcs': [
            delayed(lambda df: df['SibSp'] + df["Parch"] + 1),
            delayed(max_min_scaler),
        ]
    },
    'Ticket': {
        'funcs': [
            delayed(lambda df: df['Ticket']),
            delayed(ticket_to_class),
            delayed(one_hot(5)),
        ]
    },
    'CabinClass': {
        'funcs': [
            delayed(lambda df: df['Cabin']),
            delayed(cabin_to_class),
            delayed(one_hot(9)),
        ]
    },
    'CabinNum': {
        'funcs': [
            delayed(lambda df: df['Cabin']),
            delayed(string_int),
            delayed(max_min_scaler),
        ]
    },
    'NameLen': {
        'funcs': [
            delayed(lambda df: df['Name']),
            delayed(string_len),
            delayed(max_min_scaler),
        ]
    },
    'Name': {
        'funcs': [
            delayed(lambda df: df['Name']),
            delayed(name_to_class),
            delayed(one_hot(4)),
        ]
    },
    'Fare': {
        'funcs': [
            delayed(lambda df: df['Fare']),
            delayed(smooth_outer),
            delayed(max_min_scaler),
        ]
    },
    'Age': {
        'funcs': [
            delayed(lambda df: df['Age']),
            delayed(smooth_outer),
            delayed(max_min_scaler),
        ]
    },
    'Parch': {
        'funcs': [
            delayed(lambda df: df['Parch']),
            delayed(label_encode(range(10))),
            delayed(one_hot(10)),
        ]
    },
    'SibSp': {
        'funcs': [
            delayed(lambda df: df['SibSp']),
            delayed(label_encode(range(10))),
            delayed(one_hot(10)),
        ]
    },
    'Pclass': {
        'funcs': [
            delayed(lambda df: df['Pclass']),
            delayed(label_encode([1, 2, 3])),
            delayed(one_hot(3)),
        ]
    },
    'Sex': {
        'funcs': [
            delayed(lambda df: df['Sex']),
            delayed(label_encode(('male', 'female'))),
            delayed(one_hot(2)),
        ]
    },
    'Embarked': {
        'funcs': [
            delayed(lambda df: df['Embarked']),
            delayed(label_encode(('S', 'C', 'Q'))),
            delayed(one_hot(3)),
        ]
    },
}
fill = {'Embarked': 'S', "Age": 21, "Fare": 35.5, "Cabin": "X"}


train_df = delayed(pd.read_csv)('/store/kaggle/titanic/train.csv')
train_df = delayed(lambda df, v: df.fillna(v))(
    train_df,
    fill
)

train_x = pipe(
    preprocess_params.items(),
    map(lambda x: compose(*reversed(x[1]['funcs']))(train_df)),
    list,
)

train_y = compose(
    delayed(one_hot(2)),
    delayed(label_encode((0, 1))),
    delayed(lambda x: x['Survived'])
)(train_df)

train_dataset = delayed(TitanicDataset)(
    x=train_x,
    y=train_y
)

train_result = delayed(train)(
    model_path='/store/kaggle/titanic/model.pt',
    loss_path='/store/kaggle/titanic/train_loss.json',
    dataset=train_dataset
)

test_df = delayed(pd.read_csv)('/store/kaggle/titanic/test.csv')
test_df = delayed(lambda df, v: df.fillna(v))(
    test_df,
    fill
)
test_x = pipe(
    preprocess_params.items(),
    map(lambda x: compose(*reversed(x[1]['funcs']))(test_df)),
    list,
)

test_dataset = delayed(TitanicDataset)(test_x)

predict_result = delayed(predict)(
    delayed(lambda x: x[0])(train_result),
    test_dataset
)
submission_df = delayed(pd.DataFrame)(
    {
        'PassengerId': delayed(lambda x: x['PassengerId'])(test_df),
        "Survived": predict_result,
    }
)
submission_df = delayed(lambda x: x.set_index('PassengerId'))(submission_df)
save_submission = delayed(lambda x: x.to_csv(
    '/store/kaggle/titanic/submission.csv'))(submission_df)
