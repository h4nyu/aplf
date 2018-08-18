import torch.nn as nn
import torch
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv0 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv0_0 = nn.Conv2d(32, 32, kernel_size=3)

        self.conv1 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv1_0 = nn.Conv2d(64, 64, kernel_size=3)

        self.center0 = nn.Conv2d(64, 128, kernel_size=3)
        self.center1 = nn.Conv2d(128, 128, kernel_size=3)

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3)
        self.deconv1_0 = nn.Conv2d(128, 64, kernel_size=3)
        self.deconv1_1 = nn.Conv2d(64, 64, kernel_size=3)

        self.deconv0 = nn.ConvTranspose2d(64, 32, kernel_size=3)
        self.deconv0_0 = nn.Conv2d(64, 32, kernel_size=3)
        self.deconv0_1 = nn.Conv2d(32, 32, kernel_size=3)

        self.outconv = nn.Conv2d(32, 2, kernel_size=3)

    def forward(self, input):

        x = self.conv0(input)
        x = F.relu(x)
        x = F.layer_norm(x, x.size()[2:])
        x = self.conv0_0(x)
        conv0 = F.relu(x)

        x = self.pool(conv0)
        x = F.layer_norm(x, x.size()[2:])
        x = self.conv1(x)
        x = F.relu(x)
        x = F.layer_norm(x, x.size()[2:])
        x = self.conv1_0(x)
        conv1 = F.relu(x)

        x = self.pool(conv1)
        x = F.layer_norm(x, x.size()[2:])
        x = self.center0(x)
        x = F.relu(x)
        x = F.layer_norm(x, x.size()[2:])
        x = self.center1(x)
        x = F.relu(x)

        x = self.deconv1(x)
        x = F.relu(x)
        x = F.interpolate(x, size=conv1.size()[2:])
        x = torch.cat([x, conv1], 1)
        x = F.layer_norm(x, x.size()[2:])
        x = self.deconv1_0(x)
        x = F.relu(x)
        x = F.layer_norm(x, x.size()[2:])
        x = self.deconv1_1(x)
        x = F.relu(x)

        x = F.layer_norm(x, x.size()[2:])
        x = self.deconv0(x)
        x = F.relu(x)
        x = F.interpolate(x, size=conv0.size()[2:])
        x = torch.cat([x, conv0], 1)
        x = F.layer_norm(x, x.size()[2:])
        x = self.deconv0_0(x)
        x = F.relu(x)
        x = F.layer_norm(x, x.size()[2:])
        x = self.deconv0_1(x)
        x = F.relu(x)

        x = F.layer_norm(x, x.size()[2:])
        x = self.outconv(x)
        x = F.log_softmax(x, dim=1)
        x = F.interpolate(x, size=input.size()[2:])
        return x
