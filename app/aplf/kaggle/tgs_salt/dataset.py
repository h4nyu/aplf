from cytoolz.curried import keymap, filter, pipe, merge, map
import torch
from torch.utils.data import Dataset
import numpy as np
import glob


class TgsSaltDataset(Dataset):
    def __init__(self,
                 image_dir,
                 mark_dir):
        self.image_dir = image_dir
        self.mark_dir = mark_dir
        self.image_fns = list(glob.glob(f"{image_dir}/*.png"))
        self.mark_fns = list(glob.glob(f"{mark_dir}/*.png"))

    def __len__(self):
        return len(self.mark_fns)

    def __getitem__(self, idx):
        return idx
