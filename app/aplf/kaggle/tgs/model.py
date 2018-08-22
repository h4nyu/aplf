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

    def __init__(self, in_ch, out_ch):
        super().__init__()
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
        return x


class UpSample(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=3)
        self.deconv = nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2)
        self.conv1 = nn.Conv2d(out_ch, out_ch, kernel_size=3)
        self.se = SELayer(out_ch)
        self.activation = nn.ELU()

    def forward(self, x, bypass):
        x = torch.cat([x, F.interpolate(bypass, size=x.size()[2:])], 1)
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
        return x


class Center(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
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
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop_p = 0.2

        self.down0 = DownSample(1, 32)
        self.down1 = DownSample(32, 32)
        self.down2 = DownSample(32, 32)
        self.down3 = DownSample(32, 32)
        self.up0 = UpSample(64, 32)
        self.up1 = UpSample(64, 32)
        self.up2 = UpSample(64, 32)
        self.up3 = UpSample(64, 2)
        self.center = Center(32, 32)

    def forward(self, input):
        down0 = self.down0(input)
        down1 = self.down1(down0)
        down2 = self.down2(down1)
        down3 = self.down3(down1)
        x = self.center(down3)
        x = self.up0(x, down3)
        x = self.up1(x, down2)
        x = self.up2(x, down1)
        x = self.up3(x, down0)
        x = F.interpolate(x, size=input.size()[2:])
        return x
