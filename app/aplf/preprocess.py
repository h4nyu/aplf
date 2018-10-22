from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk, curry, merge
from sklearn.model_selection import StratifiedKFold

@curry
def kfold(dataset, n_splits, random_state=0):
    kf = StratifiedKFold(n_splits, random_state=random_state, shuffle=True)
    return list(kf.split(dataset))

