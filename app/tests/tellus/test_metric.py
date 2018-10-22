from aplf.tellus.metric import iou
import numpy as np

def test_iou():
    p = np.array([
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
    ])
    t = np.array([
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        0,
    ])
    assert iou(p, t) == 0.25

