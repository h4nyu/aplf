import numpy as np


def rl_enc(img, order='F'):
    """Convert binary mask image to run-length array or string.

    Args:
    img: image in shape [n, m]
    order: is down-then-right, i.e. Fortran(F)
    string: return in string or array

    Return:
    run-length as a string: <start[1s] length[1s] ... ...>
    """
    bytez = img.reshape(img.shape[0] * img.shape[1], order=order)
    bytez = np.concatenate([[0], bytez, [0]])
    runs = np.where(bytez[1:] != bytez[:-1])[0] + 1  # pos start at 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
