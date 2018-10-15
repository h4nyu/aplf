from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk, curry, merge
import random
import math
from sklearn.metrics import jaccard_similarity_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from skimage import io
import torch.nn.functional as F
import torch.nn as nn
import scipy
import torch
import json


def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    if not isinstance(mask_rle, str):
        return np.zeros(shape)
    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int)
                       for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def rl_enc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  # list of run lengths
    r = 0  # the current run length
    pos = 1  # count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs


def take_topk(scores, paths, top_num):
    return pipe(
        zip(scores, paths),
        lambda x: topk(top_num, x, key=lambda y: y[0]),
        map(lambda x: x[1]),
        list
    )


def is_not_reg(rle_mask):
    if(isinstance(rle_mask, str)):
        return pipe(rle_mask.split(' '),
                    len,
                    lambda x: x > 6)
    else:
        return True


def cleanup(df):
    return df[df['rle_mask'].apply(is_not_reg)]


def add_mask_size(df, shape=(101, 101)):
    df['mask_size'] = df['rle_mask']\
        .apply(lambda x: rle_decode(x, shape).sum())
    return df

def add_is_empty(df, shape=(101, 101)):
    df['is_empty'] = df['rle_mask']\
        .apply(lambda x: rle_decode(x, shape).sum() == 0)
    return df


def cut_bin(df, colunm_name, size):
    df[f'{colunm_name}_bin'] = pd.cut(df[colunm_name], size)
    return df


def groupby(df, colunm_name):
    return pipe(df.groupby(colunm_name).groups,
                map(lambda x: df[df[colunm_name] == x]),
                list)


def avarage_dfs(dfs):
    return dfs


def hflip(image):
    return image.flip([2])


def vflip(image):
    return image.flip([1])


@curry
def crop(image, start, end):
    c, w, h = image.shape
    start_h, start_w = start
    end_h, end_w = end
    image = image[:, start_w:end_w, start_h:end_h]
    image = image.view(-1, *image.shape)
    image = F.interpolate(image, mode='bilinear', size=(h, w))
    image = image.view(c, h, w)
    return image


@curry
def dump_json(path, *dicts):
    data = merge(*dicts)

    with open(path, 'w') as f:
        json.dump(data, f)
    return path


class RandomErasing(object):

    def __init__(self, p=0.5,  sl=0.01, sh=0.05, r1=1, num=1, mean=[0, 0.0, 0.0]):
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.num = num
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img
        for attempt in range(self.num):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]

        return img

@curry
def add_noise(batch_images, erase_num, erase_p):
    ramdom_erase = RandomErasing(
        num=erase_num
    )
    return pipe(
        batch_images,
        map(ramdom_erase),
        list,
        torch.stack
    )

@curry
def add_consistency_noise(batch_images, ):
    filped = batch_images.flip([3])
    return pipe(
        batch_images,
        map(ramdom_erase),
        list,
        torch.stack
    )

@curry
def kfold(dataset, n_splits, random_state=0):
    kf = KFold(n_splits, random_state=random_state)
    return list(kf.split(dataset))

def get_segment_indices(dataset, filter_indcies):
    df = dataset.df
    filtered = df.iloc[filter_indcies]
    return pipe(
        filtered[filtered['is_empty'] == False].index,
        map(df.index.get_loc),
        list
    )
