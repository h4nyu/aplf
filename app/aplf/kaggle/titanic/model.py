import torch.nn as nn
import torch.nn.functional as F


class TitanicNet(nn.Module):
    def __init__(self, input_len):
        super().__init__()
        self.fc0 = nn.Linear(input_len, 10)
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
        self.drop0 = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = self.drop0(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)
