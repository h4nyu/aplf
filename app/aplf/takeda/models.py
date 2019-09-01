import torch.nn as nn
from torch import Tensor

class ResBlock(nn.Module):
    def __init__(self,
                 in_ch:int,
                 out_ch:int,
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
            nn.Linear(in_ch, out_ch),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )
        self.activation = nn.ReLU()


    def forward(self, x:Tensor) -> Tensor:
        residual = x
        out = self.block(x)
        if self.projection:
            residual = self.projection(residual)
        out += residual
        out = self.activation(out)
        return out

class Model(nn.Module):
    def __init__(
        self,
        size_in:int,
    ) -> None:
        super().__init__()
        r = 2
        self.fc00 = ResBlock(
            size_in // (r**0),
            size_in // (r**0),
        )

        self.fc01 = ResBlock(
            size_in // (r**0),
            size_in // (r**1),
        )

        self.fc11 = ResBlock(
            size_in // (r**1),
            size_in // (r**1),
        )

        self.fc12 = ResBlock(
            size_in // (r**1),
            size_in // (r**2),
        )

        self.fc22 = ResBlock(
            size_in // (r**2),
            size_in // (r**2),
        )

        self.fc23 = ResBlock(
            size_in // (r**2),
            size_in // (r**3),
        )
        self.out = nn.Sequential(
            nn.Linear(
                size_in // (r**3),
                size_in // (r**3),
            ),
            nn.BatchNorm1d(size_in // (r**3)),
            nn.ReLU(),
            nn.Linear(
                size_in // (r**3),
                1
            ),
        )


    def forward(self, x:Tensor) -> Tensor: # type: ignore
        y = self.fc00(x)
        y = self.fc01(y)
        y = self.fc11(y)
        y = self.fc12(y)
        y = self.fc22(y)
        y = self.fc23(y)
        y = self.out(y)
        return y
