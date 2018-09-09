from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk
import torch.nn as nn
import torch
import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 res_option='A',
                 use_dropout=False
                 ):
        super().__init__()

        # uses 1x1 convolutions for downsampling

        if in_ch == out_ch:
            self.projection = None
        else:
            self.projection = nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=1,
            )
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        if self.use_dropout:
            self.dropout = nn.Dropout(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_dropout:
            out = self.dropout(out)
        if self.projection:
            residual = self.projection(x)
        out += residual
        out = self.relu2(out)
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


class DoubleConv2D(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size=3,
                 padding=1
                 ):
        super().__init__()
        self.activation = nn.ReLU()
        self.conv0 = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=padding
        )
        self.norm0 = nn.BatchNorm2d(out_ch)
        self.conv1 = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=padding
        )
        self.norm1 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.activation(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        return x


class DownSample(nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size=3,
                 padding=1,
                 ):
        super().__init__()
        self.conv0 = DoubleConv2D(
            in_ch=in_ch,
            out_ch=out_ch,
        )
        self.se = SEBlock(out_ch)
        self.conv1 = DoubleConv2D(
            in_ch=out_ch,
            out_ch=out_ch,
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.pool(x)
        return x


class UpSample(nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 other_ch=0,
                 kernel_size=3,
                 padding=1,
                 ):
        super().__init__()
        self.drop_p = 0.2
        self.deconv = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv0 = ResBlock(
            in_ch=in_ch + other_ch,
            out_ch=out_ch,
        )


    def forward(self, x, *args):
        x = self.deconv(x)
        x = torch.cat([x, *pipe(args, map(lambda l:F.interpolate(l, size=x.size()[2:])), list)], 1)
        x = self.conv0(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.ELU()
        self.down0 = DownSample(1, 64)
        self.se0 = SEBlock(64)
        self.down1 = DownSample(64, 32)
        self.se1 = SEBlock(32)
        self.down2 = DownSample(32, 16)
        self.se2 = SEBlock(16)
        self.down3 = DownSample(16, 8)
        self.se3 = SEBlock(8)
        self.down4 = DownSample(8, 4)
        self.se4 = SEBlock(4)
        self.center = DownSample(4, 4)
        self.up0 = UpSample(4, 8, 4)
        self.up_se0 = SEBlock(8)
        self.up1 = UpSample(8, 16, 8)
        self.up_se1 = SEBlock(16)
        self.up2 = UpSample(16, 32, 16)
        self.up_se2 = SEBlock(32)
        self.up3 = UpSample(32, 64, 32)
        self.up_se3 = SEBlock(64)
        self.up4 = UpSample(64, 2, 64)

    def forward(self, input):
        x = input
        down0 = self.down0(x)
        down0 = self.se0(down0)
        down1 = self.down1(down0)
        down1 = self.se1(down1)
        down2 = self.down2(down1)
        down2 = self.se2(down2)
        down3 = self.down3(down2)
        down3 = self.se3(down3)
        down4 = self.down4(down3)
        down4 = self.se4(down4)
        x = self.center(down4)
        x = self.up0(x, down4)
        x = self.up_se0(x)
        x = self.up1(x, down3)
        x = self.up_se1(x)
        x = self.up2(x, down2)
        x = self.up_se2(x)
        x = self.up3(x, down1)
        x = self.up_se3(x)
        x = self.up4(x, down0)
        x = F.interpolate(x, size=input.size()[2:])
        return x
