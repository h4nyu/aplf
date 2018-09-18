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
        self.dropout = nn.Dropout(p=0.3)

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
    def __init__(self, feature_size=32):
        super().__init__()
        self.down0 = DownSample(1, feature_size)
        self.down1 = DownSample(feature_size, feature_size // 2)
        self.down2 = DownSample(feature_size // 2, feature_size // 4)
        self.down3 = DownSample(feature_size // 4, feature_size // 8)
        self.down4 = DownSample(feature_size // 8, feature_size // 8)
        self.up0 = UpSample(feature_size // 8,
                            feature_size // 4, feature_size // 8)
        self.up1 = UpSample(feature_size // 4,
                            feature_size // 2, feature_size // 4)
        self.up2 = UpSample(feature_size // 2, feature_size, feature_size // 2)
        self.up3 = UpSample(feature_size, feature_size * 2, feature_size)
        self.ouput = nn.Conv2d(feature_size * 2, 2, kernel_size=3)

    def forward(self, x):
        x, down0 = self.down0(x)
        x, down1 = self.down1(x)
        x, down2 = self.down2(x)
        x, down3 = self.down3(x)
        _, x = self.down4(x)
        x = self.up0(x, down3)
        x = self.up1(x, down2)
        x = self.up2(x, down1)
        x = self.up3(x, down0)
        x = self.ouput(x)
        x = F.interpolate(
            x,
            mode='bilinear',
            size=(101, 101)
        )
        return x
