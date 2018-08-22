from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk
import torch.nn as nn
import torch
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ELU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DownSample(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.drop_p = 0.2
        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=3)
        self.conv1 = nn.Conv2d(out_ch, out_ch, kernel_size=3)
        self.se = SELayer(out_ch)
        self.pool = nn.MaxPool2d(2, 2)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.conv0(x)
        x = self.activation(x)
        x = F.layer_norm(x, x.size()[2:])
        x = self.conv1(x)
        x = self.activation(x)
        x = F.layer_norm(x, x.size()[2:])
        x = self.se(x)
        x = self.pool(x)
        x = F.dropout2d(x, p=self.drop_p, training=self.training)
        return x


class UpSample(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.drop_p = 0.2
        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=3)
        self.deconv = nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2)
        self.conv1 = nn.Conv2d(out_ch, out_ch, kernel_size=3)
        self.se = SELayer(out_ch)
        self.activation = nn.ELU()


    def forward(self, x, *args):
        x = torch.cat([x, *pipe(args, map(lambda l:F.interpolate(l, size=x.size()[2:])), list)], 1)
        x = self.conv0(x)
        x = self.activation(x)
        x = F.layer_norm(x, x.size()[2:])
        x = self.deconv(x)
        x = self.activation(x)
        x = F.layer_norm(x, x.size()[2:])
        x = self.conv1(x)
        x = self.activation(x)
        x = F.layer_norm(x, x.size()[2:])
        x = self.se(x)
        x = F.dropout2d(x, p=self.drop_p, training=self.training)
        return x


class Center(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.drop_p = 0.2
        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=3)
        self.deconv = nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2)
        self.conv1 = nn.Conv2d(out_ch, out_ch, kernel_size=3)
        self.se = SELayer(out_ch)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(x)
        x = F.layer_norm(x, x.size()[2:])
        x = self.deconv(x)
        x = self.activation(x)
        x = F.layer_norm(x, x.size()[2:])
        x = self.conv1(x)
        x = self.activation(x)
        x = F.layer_norm(x, x.size()[2:])
        x = self.se(x)
        x = F.dropout2d(x, p=self.drop_p, training=self.training)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down0 = DownSample(1, 64)
        self.down1 = DownSample(64, 64)
        self.center = Center(64, 64)
        self.up0 = UpSample(128, 64)
        self.up1 = UpSample(65, 64)
        self.out = nn.Conv2d(64, 2, kernel_size=5)

    def forward(self, input):
        down0 = self.down0(input)
        down1 = self.down1(down0)
        x = self.center(down1)
        x = self.up0(x, down1)
        x = self.up1(x, input)
        x = self.out(x)
        x = F.interpolate(x, size=input.size()[2:])
        return x
