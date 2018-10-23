from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk, tail, take
import torch.nn as nn
from aplf.blocks import SEBlock, ResBlock, SCSE
import torch
import torch.nn.functional as F


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
            SCSE(out_ch),
        )

        self.down = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x):
        out = self.block(x)
        conv = out
        down = self.down(conv)
        return down, conv


class SEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, r=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, int(in_ch * r)),
            nn.ELU(inplace=True),
            nn.Linear(int(in_ch * r), out_ch),
            nn.ELU(inplace=True),
        )
        self.out_ch = out_ch


    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_ch, 1, 1)
        return y


class Encorder(nn.Module):
    def __init__(self, in_ch=1, feature_size=64):
        super().__init__()
        self.down_layers = nn.ModuleList([
            DownSample(in_ch, feature_size),
            DownSample(feature_size, feature_size * 2 ** 1),
            DownSample(feature_size * 2 ** 1, feature_size * 2 ** 2),
            DownSample(feature_size * 2 ** 2, feature_size * 2 ** 3),
            DownSample(feature_size * 2 ** 3, feature_size * 2 ** 3),
        ])
    def forward(self, x):
        for layer in self.down_layers:
            x, d_out = layer(x)
        return x

class Net(nn.Module):
    def __init__(self, feature_size=64):
        super().__init__()
        self.enc = Encorder(
            in_ch=2,
            feature_size=feature_size,
        )

        self.out = SEBlock(
            in_ch=(feature_size * 2 ** 3),
            out_ch=2,
            r=1/2
        )

        self.pad = nn.ZeroPad2d(5)

    def forward(self, before, after):
        x = torch.cat([before, after], dim=1)
        _, _, h, w = x.size()
        x = F.interpolate(x, size=(2*h, 2*w))
        x = self.pad(x)
        x = self.enc(x)
        self.out(x)
        x = self.out(x).view(-1, 2)
        return x

