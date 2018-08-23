from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk
from sklearn.metrics import jaccard_similarity_score
import pandas as pd
from skimage import io


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
