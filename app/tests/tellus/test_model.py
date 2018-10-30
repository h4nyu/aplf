from aplf.tellus.models import FusionNet, MultiEncoder
import torch
import pytest


@pytest.mark.parametrize("depth, feature_size", [
    (3, 8),
])
def test_unet(depth, feature_size):
    with torch.no_grad():
        model = UNet(
            in_ch=1,
            out_ch=2,
            feature_size=feature_size,
            depth=depth,
        )
        image = torch.empty(32, 1, 40, 40)
        out = model(image)
        assert out.size() == (32, 2, 40, 40)


@pytest.mark.parametrize("depth, feature_size", [
    (2, 8),
])
def test_enc(depth, feature_size):
    with torch.no_grad():
        model = Encoder(
            in_ch=1,
            feature_size=feature_size,
            depth=depth,
            r=2,
        )
        image = torch.empty(32, 1, 80, 80)
        out = model(image)
        assert out.size() == (32, model.out_ch, 80//(2**(depth + 1)), 80//(2**(depth + 1)))


@pytest.mark.parametrize("depth, feature_size", [
    (2, 16),
])
def test_multi(depth, feature_size):
    with torch.no_grad():
        model = MultiEncoder(
            feature_size=feature_size,
        )
        parlsar_x = torch.empty(32, 2, 40, 40)
        landsat_x = torch.empty(32, 6, 4, 4)
        logit, landsat_y = model(parlsar_x)
        assert logit.size() == (32, 2)


def test_ae():
    with torch.no_grad():
        model = AE(
            in_size=(2, 40, 40),
            out_size=(2, 40, 40),
            center_out_size=(6, 4, 4),
            feature_size=8,
        )
        image = torch.empty(32, 2, 40, 40)
        pl, la = model(image)
        assert pl.size() == (32, 2, 40, 40)
        assert la.size() == (32, 6, 4, 4)


def test_fnet():
    with torch.no_grad():
        model = FusionNet(
            feature_size=16,
            resize=80,
            depth=3,
        )
        landsat_x = torch.empty(32, 2, 4, 4)
        palsar_x = torch.empty(32, 6, 40, 40)
        out = model(palsar_x, landsat_x)
        assert out.size() == (32, 2)
