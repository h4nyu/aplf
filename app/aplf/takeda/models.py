import torch.nn as nn
from torch import Tensor


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
            nn.Linear(in_ch, in_ch),
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
        r = 2
        length = 10
        self.input = nn.BatchNorm1d(size_in)
        self.layers = nn.ModuleList([
            ResBlock(
                size_in // (r ** i),
                size_in // (r ** (i+1)),
            )
            for i
            in range(length)
        ])
        self.out = nn.Sequential(
            nn.Linear(
                size_in // (r ** length),
                1
            )
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        y = self.input(x)
        for l in self.layers:
            y = l(y)
        y = self.out(y)
        return y
