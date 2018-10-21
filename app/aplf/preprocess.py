from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk, curry, merge
from sklearn.model_selection import KFold

@curry
def kfold(dataset, n_splits, random_state=0):
    kf = KFold(n_splits, random_state=random_state)
    return list(kf.split(dataset))

