import torch.nn as nn
import torch.nn.functional as F


class TgsSaltcNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.fc = nn.Linear(10 * 97 * 97, 101 * 101)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 97 * 97 * 10)
        x = F.sigmoid(self.fc(x))
        x = x.view(-1, 1,  101, 101)
        return x
