from cytoolz.curried import keymap, filter, pipe, merge, map
import torch
from torch.utils.data import Dataset


class TitanicDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, series, classes):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.series = series
        self.classes = classes
        self.class_lens = pipe(classes,
                               map(len),
                               list)

    def __len__(self):
        return len(self.series[0])

    def __getitem__(self, idx):
        return pipe(zip(self.series, self.class_lens),
                    map(lambda x: torch.eye(x[1])[x[0][idx]]),
                    list)
