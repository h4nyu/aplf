from aplf.t3tsc.models import Res34Unet
from torch import randn, empty
import torch.nn as nn

def test_res34unet() -> None:
    model = Res34Unet()
    batch_size = 1
    h = 256
    w = 256
    input_hv = randn(batch_size, 2, h, w)
    out = model(input_hv)
    assert out.shape == (batch_size, 13, h, w)



def test_loss() -> None:
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 1
    num_channels = 5
    h = 300
    w = 300
    x = randn(batch_size, num_channels, h, w)
    y = empty(batch_size, h, w).random_(num_channels).long()
    loss = loss_fn(x, y)
