from aplf.kaggle.tgs.model import UNet, SCSE, DownSample, UpSample, ResBlock, RUNet, DUNet, EUNet, HUNet, SEBlock, SHUNet
import torch
import pytest

@pytest.mark.parametrize("depth, feature_size", [
    (3, 32),
])
def test_unet(depth, feature_size):
    with torch.no_grad():
        model = UNet(
            feature_size=feature_size,
            depth=depth
        )
        in_image = torch.empty(32, 1, 101, 101)
        out_image = model(in_image)
        assert out_image.size() == (32, 2, 101, 101)

@pytest.mark.parametrize("depth, feature_size", [
    (3, 8),
])
def test_runet(depth, feature_size):
    with torch.no_grad():
        model = RUNet(
            feature_size=feature_size,
            depth=depth
        )
        in_image = torch.empty(32, 1, 101, 101)
        out_image = model(in_image)
        assert out_image.size() == (32, 2, 101, 101)

@pytest.mark.parametrize("depth, feature_size", [
    (3, 8),
])
def test_dunet(depth, feature_size):
    with torch.no_grad():
        model = DUNet(
            feature_size=feature_size,
            depth=depth
        )
        in_image = torch.empty(4, 1, 101, 101)
        out_image = model(in_image)
        assert out_image.size() == (4, 2, 101, 101)

@pytest.mark.parametrize("depth, feature_size", [
    (4, 8),
])
def test_eunet(depth, feature_size):
    with torch.no_grad():
        model = EUNet(
            feature_size=feature_size,
            depth=depth
        )
        in_image = torch.empty(4, 1, 101, 101)
        out_image = model(in_image)
        assert out_image.size() == (4, 2, 101, 101)

@pytest.mark.parametrize("feature_size", [
    64,
])
def test_hunet(feature_size):
    with torch.no_grad():
        model = HUNet(
            feature_size=feature_size,
        )
        in_image = torch.empty(4, 1, 101, 101)
        out_image, center = model(in_image)
        assert out_image.size() == (4, 2, 101, 101)


@pytest.mark.parametrize("feature_size", [
    64,
])
def test_shunet(feature_size):
    with torch.no_grad():
        model = SHUNet(
            feature_size=feature_size,
        )
        in_image = torch.empty(4, 1, 202, 202)
        out_image, center, before_out = model(in_image)

        assert out_image.size() == (4, 2, 101, 101)

def test_scse():
    model = SCSE(32, 2)
    in_image = torch.empty(32, 32, 101, 101)
    output = model(in_image)
    assert output.size() == (32, 32, 101, 101)



def test_res_block():
    model = ResBlock(32, 16)
    in_image = torch.empty(32, 32, 101, 101)
    output = model(in_image)
    assert output.size() == (32, 16, 101, 101)

def test_se_block():
    model = SEBlock(32, 16)
    in_image = torch.empty(32, 32, 101, 101)
    output = model(in_image)
    assert output.size() == (32, 16, 1, 1)


def test_downsample():
    model = DownSample(1, 32, kernel_size=3)
    in_image = torch.empty(32, 1, 101, 101)
    pooled, conved = model(in_image)
    assert pooled.size() == (32, 32, 50, 50)
    assert conved.size() == (32, 32, 101, 101)


def test_upsample():
    model = UpSample(16, 32, 32)
    in_image = torch.empty(32, 16, 48, 48)
    bypass_image = torch.empty(32, 32, 96, 96)
    output = model(in_image, bypass_image)
    assert output.size() == (32, 32, 96, 96)
