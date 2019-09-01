import torch.nn as nn
from torch import Tensor

class Model(nn.Module):
    def __init__(
        self,
        size_in:int,
    ) -> None:
        super().__init__()
        r = 2
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(size_in),
            nn.Linear(size_in, size_in//r),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(size_in//2),
            nn.Linear(size_in//2, size_in//4),
            nn.ReLU(inplace=True),
        )

        self.fc3 = nn.Sequential(
            nn.BatchNorm1d(size_in//4),
            nn.Linear(size_in//4, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x:Tensor) -> Tensor: # type: ignore
        y = self.fc1(x)
        y = self.fc2(y)
        y = self.fc3(y)
        return y
