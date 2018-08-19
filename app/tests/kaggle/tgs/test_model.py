from aplf.kaggle.tgs.model import UNet, TinyUNet
import torch


def test_model():
    model = UNet()
    print(model)
    in_image = torch.empty(32, 1, 101, 101)
    out_image = model(in_image)
    assert out_image.size() == (32, 2, 101, 101)


def test_boost():
    model = UNet(in_ch=3)
    in_image = torch.empty(32, 1, 101, 101)
    segment = torch.empty(32, 2, 101, 101)
    out_image = model(in_image, segment)
    assert out_image.size() == (32, 2, 101, 101)


def test_tiny():
    model = TinyUNet()
    print(model)
    in_image = torch.empty(32, 1, 101, 101)
    out_image = model(in_image)
    assert out_image.size() == (32, 2, 101, 101)
