from aplf.kaggle.tgs.model import UNet, SELayer, DownSample, UpSample
import torch


def test_model():
    model = UNet()
    print(model)
    in_image = torch.empty(32, 1, 101, 101)
    out_image = model(in_image)
    assert out_image.size() == (32, 2, 101, 101)


def test_se_layer():
    model = SELayer(32, 2)
    in_image = torch.empty(32, 32, 101, 101)
    output = model(in_image)
    assert output.size() == (32, 32, 101, 101)


def test_downsample():
    model = DownSample(1, 32, kernel_size=3)
    in_image = torch.empty(32, 1, 101, 101)
    output = model(in_image)
    assert output.size() == (32, 32, 48, 48)


def test_upsample():
    model = UpSample(16, 32, 32, kernel_size=3)
    in_image = torch.empty(32, 16, 48, 48)
    bypass_image = torch.empty(32, 32, 96, 96)
    output = model(in_image, bypass_image)
    assert output.size() == (32, 32, 92, 92)
