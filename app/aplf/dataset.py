from torch.utils.data import Dataset
from cytoolz.curried import pipe, map, take


class DummyDataset(Dataset):

    def __init__(self,
                 matrix,
                 window_size,
                 annomalies=[],
                 transform=None):
        self.matrix = matrix
        self.annomalies = annomalies
        self.window_size = window_size

    def __len__(self):
        return self.matrix.shape[0] + self.window_size

    def __getitem__(self, idx):
        label = pipe(self.annomalies,
                     map(lambda x: x[0] <= idx and idx <= x[1]),
                     any)

        return (self.matrix[idx: idx + self.window_size], label)
