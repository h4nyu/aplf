from cytoolz.curried import keymap, filter, pipe, merge, map
from toolz import curry
from sklearn import preprocessing
import numpy as np
import pandas as pd
from dask import delayed
import re


@curry
def label_encode(series):
    le = preprocessing.LabelEncoder()
    return le.fit_transform(series)


def max_min_scaler(series):
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(series.values.reshape(-1, 1))


def standard_scaler(series):
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(series.values.reshape(-1, 1))


def string_len(series):
    return series.apply(len)


@curry
def clean_up_rare(series, callback, min_stat=10):
    rare_values = series.value_counts() < min_stat
    return series.apply(lambda x: callback(rare_values, x))


@curry
def one_hot(series):
    max_value = series.max() + 1
    eye = np.eye(max_value)
    return pipe(series, map(lambda x: eye[x]), list)


def string_int(series):
    def _int(x):
        matched = re.findall(r'\d+', x)
        return int(matched[0]) if len(matched) > 0 else 0
    return series.apply(_int)


def smooth_outer(series):
    q1 = np.percentile(series, 25)
    q3 = np.percentile(series, 75)
    outlier_step = 2 * (q3 - q1)

    def _outer(x):
        if x < (q1 - outlier_step):
            return q1
        elif x > (q3 + outlier_step):
            return q3
        else:
            return x

    return series.apply(_outer)


def cabin_to_class(series):
    def _contain(x):
        if "A" in x:
            return 1
        elif "B" in x:
            return 2
        elif "C" in x:
            return 3
        elif "D" in x:
            return 4
        elif "E" in x:
            return 5
        elif "F" in x:
            return 6
        elif "G" in x:
            return 7
        elif "T" in x:
            return 8
        else:
            return 0

    return series.apply(_contain)


def ticket_to_class(series):
    def _contain(x):
        if "A5" in x:
            return 1
        elif "PC" in x:
            return 2
        elif "ST" in x:
            return 3
        elif "X" in x:
            return 4
        else:
            return 0

    return series.apply(_contain)


def name_to_class(series):
    def _contain(x):
        if "Master" in x:
            return 1
        elif "Mme" in x:
            return 2
        elif "Mrs" in x:
            return 2
        elif "Ms" in x:
            return 2
        elif "Mr" in x:
            return 3
        else:
            return 0

    return series.apply(_contain)
