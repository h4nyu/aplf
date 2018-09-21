from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk
import torch.nn as nn
import torch
import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 use_dropout=True
                 ):
        super().__init__()
        self.use_dropout = use_dropout
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
                padding=1
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_ch),
        )
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3, inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        if self.projection:
            residual = self.projection(residual)
        if self.use_dropout:
            out = self.dropout(out)
        out += residual
        out = self.activation(out)
        return out


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=2 / 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel * reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel * reduction), channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DownSample(nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size=3,
                 padding=1,
                 ):
        super().__init__()

        self.block = nn.Sequential(
            ResBlock(
                in_ch=in_ch,
                out_ch=out_ch,
            ),
            SEBlock(out_ch),
            ResBlock(
                in_ch=out_ch,
                out_ch=out_ch,
            ),
            SEBlock(out_ch),
            ResBlock(
                in_ch=out_ch,
                out_ch=out_ch,
            ),
            SEBlock(out_ch),
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
                 other_ch,
                 kernel_size=3,
                 padding=1,
                 ):
        super().__init__()
        self.block = nn.Sequential(
            ResBlock(
                in_ch + other_ch,
                out_ch,
            ),
            SEBlock(out_ch),
        )

    def forward(self, x, other):
        x = F.interpolate(x, mode='bilinear', size=other.size()[2:])
        x = torch.cat([x, other], 1)
        x = self.block(x)
        return x


class UNet(nn.Module):
    def __init__(self, feature_size=8, depth=3):
        super().__init__()
        self.down_layers = nn.ModuleList([
            DownSample(1, feature_size),
            *pipe(
                range(depth),
                map(lambda x: DownSample(
                    feature_size * (2 ** x),
                    feature_size * (2 ** (x + 1)),
                )),
                list,
            )
        ])

        self.center = DownSample(
            feature_size * (2 ** depth),
            feature_size * (2 ** depth),
        )

        self.up_layers = nn.ModuleList([
            *pipe(
                range(depth),
                reversed,
                map(lambda x: UpSample(
                    feature_size * (2 ** (x + 1)),
                    feature_size * (2 ** x),
                    feature_size * (2 ** (x + 1)),
                )),
                list,
            ),
            UpSample(feature_size, feature_size, feature_size)
        ])
        self._output = nn.Conv2d(feature_size, 2, kernel_size=3)

    def forward(self, x):
        # down samples
        d_outs = []
        for layer in self.down_layers:
            x, d_out = layer(x)
            d_outs.append(d_out)

        # center
        _, x = self.center(x)

        # up samples
        for layer, d_out in zip(self.up_layers, reversed(d_outs)):
            x = layer(x, d_out)

        x = self._output(x)
        # output
        x = F.interpolate(
            x,
            mode='bilinear',
            size=(101, 101)
        )
        return x
