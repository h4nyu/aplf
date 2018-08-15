import torch.nn as nn
import torch
import torch.nn.functional as F


class TgsSaltcNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm0 = nn.LayerNorm((1, 101, 101))
        self.conv0 = nn.Conv2d(1, 32, kernel_size=3)
        self.norm1 = nn.LayerNorm((32, 99, 99))
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3)
        self.norm2 = nn.LayerNorm((32, 97, 97))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)

        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3)
        self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3)
        self.deconv0 = nn.ConvTranspose2d(32, 2, kernel_size=3)

    def forward(self, x):
        x = self.norm0(x)
        x = self.conv0(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = self.deconv2(x)
        x = F.relu(x)
        x = self.deconv1(x)
        x = F.relu(x)
        x = self.deconv0(x)
        x = F.log_softmax(x, dim=1)
        return x
