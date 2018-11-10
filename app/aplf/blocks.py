from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk, tail, take
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, r=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//r, kernel_size=1),
            nn.ELU(inplace=True),
            nn.Conv2d(in_ch//r, out_ch, kernel_size=1),
            nn.Sigmoid()
        )
        self.out_ch = out_ch

    def forward(self, x):
        avg_out = self.avg_fc(self.avg_pool(x))
        max_out = self.max_fc(self.max_pool(x))
        return avg_out + max_out


class CSE(nn.Module):
    def __init__(self, in_ch, r=1):
        super().__init__()
        self.se = SEBlock(in_ch, in_ch, r=r)

    def forward(self, x):
        y = self.se(x)
        return x * y


class SSE(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x
        x = self.conv(x)
        x = F.sigmoid(x)
        x = input_x * x

        return x


class SCSE(nn.Module):
    def __init__(self, in_ch, r=2):
        super(SCSE, self).__init__()

        self.cSE = CSE(in_ch, r)
        self.sSE = SSE(in_ch)

    def forward(self, x):
        cSE = self.cSE(x)
        sSE = self.sSE(x)
        x = cSE + sSE
        return x


class ResBlock(nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 ):
        super().__init__()
        if in_ch == out_ch:
            self.projection = None
        else:
            self.projection = nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=1,
            )
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=2 - (out_ch % 2),
            ),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.BatchNorm2d(out_ch),
            SCSE(out_ch),
        )
        self.activation = nn.ELU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        if self.projection:
            residual = self.projection(residual)
        out += residual
        out = self.activation(out)
        return out


class DownSample(nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size=3,
                 padding=1,
                 ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.block = nn.Sequential(
            ResBlock(
                in_ch=in_ch,
                out_ch=out_ch,
            ),
            ResBlock(
                in_ch=out_ch,
                out_ch=out_ch,
            ),
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = self.block(x)
        conv = out
        down = self.pool(conv)
        return down, conv


class UpSample(nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size=3,
                 padding=1,
                 ):
        super().__init__()
        self.block = nn.Sequential(
            ResBlock(
                in_ch,
                out_ch,
            ),
        )

    def forward(self, x, others, size):
        out = pipe(
            [x, *others],
            map(lambda x: F.interpolate(x, mode='bilinear', size=size)),
            list
        )
        out = torch.cat([*out], 1)
        out = self.block(out)
        return out
