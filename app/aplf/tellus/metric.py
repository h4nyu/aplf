import numpy as np


def iou(p, t):
    intersection = np.logical_and(t, p)
    if np.sum(intersection) == 0:
        return 0

    union = np.logical_or(t, p)
    iou = np.sum(intersection > 0) / np.sum(union > 0)
    return iou
