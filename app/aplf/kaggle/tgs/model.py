import torch.nn as nn
import torch
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.drop_p = 0.2
        self.pool = nn.MaxPool2d(2, 2)

        self.conv0 = nn.Conv2d(in_ch, 32, kernel_size=3)
        self.conv0_0 = nn.Conv2d(32, 32, kernel_size=3)

        self.conv1 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv1_0 = nn.Conv2d(64, 64, kernel_size=3)

        self.center0 = nn.Conv2d(64, 128, kernel_size=3)
        self.center1 = nn.Conv2d(128, 128, kernel_size=3)

        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, stride=3)
        self.deconv1_0 = nn.Conv2d(128, 64, kernel_size=3)
        self.deconv1_1 = nn.Conv2d(64, 64, kernel_size=3)

        self.deconv0 = nn.ConvTranspose2d(64, 32, 3, stride=3)
        self.deconv0_0 = nn.Conv2d(64, 32, kernel_size=3)
        self.deconv0_1 = nn.Conv2d(32, 32, kernel_size=3)

        self.outconv = nn.Conv2d(32, 2, kernel_size=3)

    def forward(self, input, segment=None):
        x = None
        if segment is not None:
            x = torch.cat([input, segment], 1)
        else:
            x = input

        x = self.conv0(x)
        x = F.relu(x)
        x = F.layer_norm(x, x.size()[2:])
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.conv0_0(x)
        conv0 = F.relu(x)

        x = self.pool(conv0)
        x = F.layer_norm(x, x.size()[2:])
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.layer_norm(x, x.size()[2:])
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.conv1_0(x)
        conv1 = F.relu(x)

        x = self.pool(conv1)
        x = F.layer_norm(x, x.size()[2:])
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.center0(x)
        x = F.relu(x)
        x = F.layer_norm(x, x.size()[2:])
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.center1(x)
        x = F.relu(x)

        x = self.deconv1(x)
        x = F.relu(x)
        x = F.interpolate(x, size=conv1.size()[2:])
        x = torch.cat([x, conv1], 1)
        x = F.layer_norm(x, x.size()[2:])
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.deconv1_0(x)
        x = F.relu(x)
        x = F.layer_norm(x, x.size()[2:])
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.deconv1_1(x)
        x = F.relu(x)

        x = F.layer_norm(x, x.size()[2:])
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.deconv0(x)
        x = F.relu(x)
        x = F.interpolate(x, size=conv0.size()[2:])
        x = torch.cat([x, conv0], 1)
        x = F.layer_norm(x, x.size()[2:])
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.deconv0_0(x)
        x = F.relu(x)
        x = F.layer_norm(x, x.size()[2:])
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.deconv0_1(x)
        x = F.relu(x)

        x = F.layer_norm(x, x.size()[2:])
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.outconv(x)
        x = F.interpolate(x, size=input.size()[2:])
        return x


class TinyUNet(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.drop_p = 0.2
        self.pool = nn.MaxPool2d(2, 2)

        self.conv0 = nn.Conv2d(in_ch, 32, kernel_size=3)
        self.conv0_0 = nn.Conv2d(32, 32, kernel_size=3)

        self.center0 = nn.Conv2d(32, 64, kernel_size=3)
        self.center1 = nn.Conv2d(64, 64, kernel_size=3)

        self.deconv0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.deconv0_1 = nn.Conv2d(64, 32, kernel_size=3)

        self.outconv = nn.Conv2d(32, 2, kernel_size=3)

    def forward(self, input):

        x = self.conv0(input)
        x = F.relu(x)
        x = F.layer_norm(x, x.size()[2:])
        conv0 = F.dropout(x, p=self.drop_p, training=self.training)

        x = self.pool(conv0)
        x = F.layer_norm(x, x.size()[2:])
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.center0(x)
        x = F.relu(x)
        x = F.layer_norm(x, x.size()[2:])
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.center1(x)
        x = F.relu(x)

        x = F.layer_norm(x, x.size()[2:])
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.deconv0(x)
        x = F.relu(x)
        x = F.interpolate(x, size=conv0.size()[2:])
        x = torch.cat([x, conv0], 1)
        x = F.layer_norm(x, x.size()[2:])
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.deconv0_1(x)
        x = F.relu(x)

        x = F.layer_norm(x, x.size()[2:])
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.outconv(x)
        x = F.interpolate(x, size=input.size()[2:])
        return x
