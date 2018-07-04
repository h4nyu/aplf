from cytoolz.curried import keymap, filter, pipe, merge, map
from sklearn import preprocessing
import numpy as np
from dask import delayed


@delayed
def label_encode(series, classes):
    le = preprocessing.LabelEncoder()
    le.fit(classes)
    return le.transform(series)


@delayed
def max_min_scaler(series):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(series.values.reshape(-1, 1))
    return x_scaled


@delayed
def one_hot(series, class_len):
    return pipe(series,
                map(lambda x: np.eye(class_len)[x]),
                list,
                np.array)
