import numpy as np


def iou(p, t):
    if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
        return 0
    if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
        return 0
    if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
        return 1

    intersection = np.logical_and(t, p)
    union = np.logical_or(t, p)
    iou = np.sum(intersection > 0) / np.sum(union > 0)
    thresholds = np.arange(0.5, 1, 0.05)
    s = []
    for thresh in thresholds:
        s.append(iou > thresh)
    return np.mean(s)
