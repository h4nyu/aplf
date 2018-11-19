from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk, curry, merge
from sklearn.model_selection import StratifiedKFold
import random
import math
from sklearn.metrics import jaccard_similarity_score
import numpy as np
import pandas as pd
from skimage import io
import torch.nn.functional as f
import torch.nn as nn
import scipy
import torch
import json



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

