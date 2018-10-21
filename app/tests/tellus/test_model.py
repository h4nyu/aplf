from aplf.tellus.model import Net
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

