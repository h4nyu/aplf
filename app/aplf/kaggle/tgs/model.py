from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk
import torch.nn as nn
import torch
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
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

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.drop_p = 0.2
        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size)
        self.conv1 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size)
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

    def __init__(self, in_ch, out_ch, other_ch=0, kernel_size=3):
        super().__init__()
        self.drop_p = 0.2
        self.deconv = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv0 = nn.Conv2d(in_ch + other_ch,
                               out_ch,
                               kernel_size=kernel_size)
        self.conv1 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size)
        self.se = SELayer(out_ch)
        self.activation = nn.ELU()

    def forward(self, x, *args):
        x = self.deconv(x)
        x = self.activation(x)
        x = F.layer_norm(x, x.size()[2:])
        x = torch.cat([x, *args], 1)
        x = self.conv0(x)
        x = self.activation(x)
        x = F.layer_norm(x, x.size()[2:])
        x = self.conv1(x)
        x = self.activation(x)
        x = F.layer_norm(x, x.size()[2:])
        x = self.se(x)
        x = F.dropout2d(x, p=self.drop_p, training=self.training)
        return x


class Center(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super().__init__()
        self.drop_p = 0.2
        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size)
        self.deconv = nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2)
        self.conv1 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size)
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
        self.activation = nn.ELU()
        self.down0 = DownSample(1, 48)
        self.down1 = DownSample(48, 16)
        self.center = Center(16, 8)
        self.up0 = UpSample(8, 32, 48)
        self.up1 = UpSample(33, 2)

    def forward(self, input):
        x = input
        down0 = self.down0(x)
        down1 = self.down1(down0)
        x = self.center(down1)
        print(x.size())
        print(down0.size())
        x = self.up0(x, F.interpolate(down0, x.size()[2:]))
        print(x.size())
        x = self.up1(x, input)
        return x
