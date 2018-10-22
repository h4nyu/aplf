from aplf.tellus.model import Net, DownSample
import torch
import pytest


@pytest.mark.parametrize("depth, feature_size", [
    (3, 32),
])
def test_unet(depth, feature_size):
    with torch.no_grad():
        model = Net(
            feature_size=feature_size,
        )
        before = torch.empty(32, 1, 40, 40)
        after = torch.empty(32, 1, 40, 40)
        out_image = model(before, after)
        assert out_image.size() == (32, 2)


def test_down():
    with torch.no_grad():
        model = DownSample(
            in_ch=10,
            out_ch=4,
        )
        image = torch.empty(32, 10, 40, 40)
        down_image, out_image = model(image)
        assert down_image.size() == (32, 4, 20, 20)
        assert out_image.size() == (32, 4, 40, 40)
