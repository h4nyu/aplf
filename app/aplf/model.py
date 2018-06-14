from datetime import datetime
from dateutil import parser
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from cytoolz.curried import pipe, map, take


class Autoencoder(nn.Module):

    def __init__(self, window_size):
        super().__init__()
        self.conv0 = nn.Conv1d(1, 20, kernel_size=3)
        before_fc_size = int((window_size - 2) * 20 / 2)
        self.fc0 = nn.Linear(before_fc_size, 60)
        self.fc1 = nn.Linear(60, 20)
        self.defc0 = nn.Linear(20, 60)
        self.defc1 = nn.Linear(60, before_fc_size)
        self.deconv0 = nn.ConvTranspose1d(20, 1, kernel_size=3)
        self.pool0 = nn.MaxPool1d(2, 2, return_indices=True)
        self.unpool0 = nn.MaxUnpool1d(2, 2)

    def forward(self, x):
        conv0 = self.conv0(x)
        x, pool0_indices = self.pool0(conv0)
        x = F.tanh(x)
        before_fc = F.dropout(x)
        x = x.view(-1, before_fc.size(1) * before_fc.size(2))
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.defc0(x))
        x = F.relu(self.defc1(x))
        x = x.view(-1, before_fc.size(1), before_fc.size(2))
        x = self.unpool0(x, indices=pool0_indices, output_size=conv0.size())
        x = self.deconv0(x)
        x = x.view(-1, x.size(1), x.size(2))
        return x
