#!/ usr / bin / env python
# -*- coding: utf-8 -*-
from cytoolz.curried import keymap, filter, pipe, merge, map, compose, valmap
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
    ticket_to_class,
    clean_up_rare
)
from .train import train
from .predict import predict, evaluate


base_dir = '/store/titanic'


preprocess_params = {
    'FareBin': {
        'funcs': [
            delayed(lambda df: df['Fare']),
            delayed(lambda s: s.fillna(s.median())),
            delayed(lambda s: pd.qcut(s, 4)),
            delayed(label_encode),
            delayed(one_hot),
        ]
    },
    'AgeBin': {
        'funcs': [
            delayed(lambda df: df['Age']),
            delayed(lambda s: s.fillna(s.mean())),
            delayed(lambda s: pd.cut(s, 5)),
            delayed(label_encode),
            delayed(one_hot),
        ]
    },

    'isAlone': {
        'funcs': [
            delayed(lambda df: df['SibSp'] + df['Parch'] + 1),
            delayed(lambda s: s == 1),
            delayed(label_encode),
            delayed(one_hot),
        ]
    },
    'TitleCode': {
        'funcs': [
            delayed(lambda df: df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]),
            lambda x: delayed(clean_up_rare)(
                x,
                lambda rares, x: "Misc"  if rares.loc[x] == True else x
            ),
            delayed(label_encode),
            delayed(one_hot),
        ]
    },
    'Pclass': {
        'funcs': [
            delayed(lambda df: df['Pclass']),
            delayed(one_hot),
        ]
    },
    'SexCode': {
        'funcs': [
            delayed(lambda df: df['Sex']),
            delayed(label_encode),
            delayed(one_hot),
        ]
    },
    'Embarked': {
        'funcs': [
            delayed(lambda df: df['Embarked']),
            delayed(lambda s:s.fillna(s.mode()[0])),
            delayed(label_encode),
            delayed(one_hot),
        ]
    },
    'Survived': {
        'funcs': [
            delayed(lambda df: df['Survived']),
            delayed(one_hot),
        ]
    },
}


train_df = delayed(pd.read_csv)('/store/kaggle/titanic/train.csv')

preprocessed_train_df = pipe(
    preprocess_params,
    valmap(lambda x: compose(*reversed(x['funcs']))(train_df)),
    delayed(pd.DataFrame),
)


train_dataset = delayed(TitanicDataset)(
    df=preprocessed_train_df,
)

#  train_y = compose(
#      delayed(one_hot(2)),
#      delayed(label_encode((0, 1))),
#      delayed(lambda x: x['Survived'])
#  )(train_df)
#
#
#  train_result = delayed(train)(
#      model_path='/store/kaggle/titanic/model.pt',
#      loss_path='/store/kaggle/titanic/train_loss.json',
#      dataset=train_dataset
#  )

test_df = delayed(pd.read_csv)('/store/kaggle/titanic/test.csv')
test_x = pipe(
    preprocess_params.items(),
    map(lambda x: compose(*reversed(x[1]['funcs']))(test_df)),
    list,
)
#
#  test_dataset = delayed(TitanicDataset)(test_x)
#
#  predict_result = delayed(predict)(
#      delayed(lambda x: x[0])(train_result),
#      test_dataset
#  )
#  submission_df = delayed(pd.DataFrame)(
#      {
#          'PassengerId': delayed(lambda x: x['PassengerId'])(test_df),
#          "Survived": predict_result,
#      }
#  )
#  submission_df = delayed(lambda x: x.set_index('PassengerId'))(submission_df)
#  save_submission = delayed(lambda x: x.to_csv(
#      '/store/kaggle/titanic/submission.csv'))(submission_df)
