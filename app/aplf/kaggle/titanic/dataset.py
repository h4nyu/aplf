from cytoolz.curried import keymap, filter, pipe, merge, map
import torch
from torch.utils.data import Dataset


class TitanicDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,
                 x_series,
                 x_classes,
                 y_series,
                 y_classes):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.x_series = x_series
        self.x_classes = x_classes
        self.x_class_lens = pipe(x_classes,
                                 map(len),
                                 list)
        self.y_series = y_series
        self.y_classes = y_classes
        self.y_class_lens = pipe(y_classes,
                                 map(len),
                                 list)
        self.x_len = sum(self.x_class_lens)
        self.y_len = sum(self.y_class_lens)

    def __len__(self):
        return len(self.x_series[0])

    def __getitem__(self, idx):
        x = pipe(zip(self.x_series, self.x_class_lens),
                 map(lambda x: torch.eye(x[1])[x[0][idx]]),
                 list,
                 torch.cat)
        return x, self.y_series[0][idx]
