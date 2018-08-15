import torch.nn as nn
import torch.nn.functional as F


class TgsSaltcNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3)
        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.deconv1(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = F.relu(x)
        x = self.deconv3(x)
        x = F.sigmoid(x)
        return x
