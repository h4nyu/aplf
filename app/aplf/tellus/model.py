from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk, tail, take
import torch.nn as nn
from aplf.blocks import DownSample, SEBlock
import torch
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, feature_size=64):
        super().__init__()
        self.down_layers = nn.ModuleList([
            DownSample(2, feature_size),
            DownSample(feature_size, feature_size * 2 ** 1),
            DownSample(feature_size * 2 ** 1, feature_size * 2 ** 2),
            DownSample(feature_size * 2 ** 2, feature_size * 2 ** 3),
            DownSample(feature_size * 2 ** 3, feature_size * 2 ** 3),
        ])

        self.out = SEBlock(
            in_ch=feature_size * 2 ** 3,
            out_ch=2,
        )

        self.pad = nn.ZeroPad2d(5)

    def forward(self, before, after):
        x = torch.cat([before, after], 1)
        x = F.interpolate(x, mode='bilinear', size=(108, 108))
        x = self.pad(x)

        d_outs = []
        for layer in self.down_layers:
            x, d_out = layer(x)
            d_outs.append(d_out)

        x = self.out(x).view(-1, 2)
        return x
