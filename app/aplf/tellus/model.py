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


class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, feature_size=64):
        super().__init__()
        self.down_layers = nn.ModuleList([
            DownSample(in_ch, feature_size),
            DownSample(feature_size, feature_size * 2 ** 1),
            DownSample(feature_size * 2 ** 1, feature_size * 2 ** 2),
            DownSample(feature_size * 2 ** 2, out_ch),
        ])

    def forward(self, x):
        d_outs = []
        for layer in self.down_layers:
            x, d_out = layer(x)
        return x


class Fc(nn.Module):
    def __init__(self, in_ch, out_ch, feature_size=64, r=2):
        super().__init__()
        self.before_pool = ResBlock(
            in_ch=in_ch,
            out_ch=in_ch//r,
        ),
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch//r//r),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch//r//r, out_ch),
        )
        self.out_ch = out_ch

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_ch, 1, 1)
        return y


class Net(nn.Module):
    def __init__(self,
                 feature_size=64,
                 resize=120,
                 pad=4,
                 ):
        super().__init__()
        self.resize = resize

        self.seg_enc = Encoder(
            in_ch=2,
            out_ch=feature_size * 2 ** 3,
            feature_size=feature_size,
        )

        self.rgb_enc = Encoder(
            in_ch=1,
            out_ch=3,
            feature_size=feature_size,
        )

        self.out = Fc(
            in_ch=feature_size * 2 ** 3 + 6,
            out_ch=2
        )

        self.pad = nn.ZeroPad2d(pad)

    def forward(self, b_x, a_x):
        b_x = F.interpolate(b_x, mode='bilinear', size=(self.resize, self.resize))
        a_x = F.interpolate(a_x, mode='bilinear', size=(self.resize, self.resize))
        b_x = self.pad(b_x)
        a_x = self.pad(a_x)

        x = torch.cat([b_x, a_x], dim=1)
        x = self.seg_enc(x)

        b_rgb = self.rgb_enc(b_x)
        a_rgb = self.rgb_enc(a_x)

        x = torch.cat([x, b_rgb, a_rgb], dim=1)
        self.out(x)
        x = self.out(x).view(-1, 2)
        b_rgb = F.interpolate(b_rgb, mode='bilinear', size=(4, 4))
        a_rgb = F.interpolate(a_rgb, mode='bilinear', size=(4, 4))
        return x, b_rgb, a_rgb
