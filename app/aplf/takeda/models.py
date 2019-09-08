import torch.nn as nn
from torch import Tensor
from torch.nn.functional import relu


class ResBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 ) -> None:
        super().__init__()
        if in_ch == out_ch:
            self.projection = None
        else:
            self.projection = nn.Linear(
                in_ch,
                out_ch,
            )
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.block(x)
        out += residual
        if self.projection:
            out = self.projection(out)
        return out


class Model(nn.Module):
    def __init__(
        self,
        size_in: int,
    ) -> None:
        super().__init__()
        r = 4
        self.input = nn.Sequential(
            nn.BatchNorm1d(size_in)
        )
        self.fc0 = ResBlock(
            size_in // (r**0),
            size_in // (r**1),
        )

        self.fc1 = ResBlock(
            size_in // (r**1),
            size_in // (r**2),
        )

        self.fc2 = ResBlock(
            size_in // (r**2),
            size_in // (r**3),
        )

        self.out = nn.Sequential(
            nn.Linear(
                size_in // (r**3),
                1,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        y = self.input(x)
        y = self.fc0(x)
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.out(y)
        return y
