import torch.nn as nn
import torch
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.norm0 = nn.LayerNorm((1, 101, 101))
        self.conv0 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv1 = nn.Conv2d(32, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3)

        self.fc0 = nn.Linear(8 * 10 * 10, 8 * 10 * 10)

        self.deconv2 = nn.ConvTranspose2d(16, 16, kernel_size=3)
        self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3)
        self.deconv0 = nn.ConvTranspose2d(64, 2, kernel_size=3)

    def forward(self, input):

        x = self.conv0(input)
        x = self.pool(x)
        conv0 = F.relu(x)

        x = self.conv1(conv0)
        x = self.pool(x)
        conv1 = F.relu(x)

        x = self.conv2(conv1)
        x = self.pool(x)
        conv2 = F.relu(x)

        x = conv2.view(-1, 8 * 10 * 10)
        x = self.fc0(x)
        x = F.relu(x)
        center = x.view(-1, 8, 10, 10)

        x = torch.cat([center, conv2], 1)
        x = self.deconv2(x)
        x = F.interpolate(x, size=conv1.size()[2:])
        dec2 = F.relu(x)

        x = torch.cat([dec2, conv1], 1)
        x = self.deconv1(x)
        x = F.interpolate(x, size=conv0.size()[2:])
        dec1 = F.relu(x)

        x = torch.cat([dec1, conv0], 1)
        x = self.deconv0(x)
        x = F.interpolate(x, size=input.size()[2:])
        x = F.log_softmax(x, dim=1)
        return x
