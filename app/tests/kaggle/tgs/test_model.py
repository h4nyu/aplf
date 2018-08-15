from aplf.kaggle.tgs.model import TgsSaltcNet
import torch


def test_model():
    model = TgsSaltcNet()
    in_image = torch.empty(32, 1, 101, 101)
    out_image = model(in_image)
    assert out_image.size() == (32, 2, 101, 101)
