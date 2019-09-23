from torch import nn
from torch import Tensor
from torch.nn.functional import relu, dropout

class Swish(nn.Module):
    def forward(self, x: Tensor) -> Tensor: # type: ignore
        return x * x.sigmoid()

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
            Swish(),
            nn.Linear(in_ch, in_ch // 2),
            nn.BatchNorm1d(in_ch // 2),
            Swish(),
            nn.Linear(in_ch // 2, in_ch),
            nn.BatchNorm1d(in_ch),
            Swish(),
        )

    def forward(self, x: Tensor) -> Tensor:# type: ignore
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
        r = 100
        self.input = nn.BatchNorm1d(size_in)
        self.fc0 = ResBlock(
            size_in // (r**0),
            size_in // (r**1),
        )
        self.out = nn.Sequential(
            nn.Linear(
                size_in // (r**1),
                1
            ),
        )


    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        y = self.input(x)
        y = self.fc0(y)
        y = self.out(y)
        return y
