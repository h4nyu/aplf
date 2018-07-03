import torch.nn as nn
import torch.nn.functional as F


class TitanicNet(nn.Module):
    def __init__(self, input_len, output_len):
        super().__init__()
        self.fc0 = nn.Linear(input_len, 4)
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, output_len)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
