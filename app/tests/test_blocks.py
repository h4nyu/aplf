from aplf.blocks import ChannelAttention, SpatialAttention, CBAM
import torch
import pytest


def test_ca():
    with torch.no_grad():
        model = ChannelAttention(
            in_ch=64,
            out_ch=32,
            r=16,
        )
        image = torch.empty(32, 64, 40, 40)
        out = model(image)
        assert out.size() == (32, 32, 1, 1)

def test_sa():
    with torch.no_grad():
        model = SpatialAttention()
        image = torch.empty(32, 64, 40, 40)
        out = model(image)
        assert out.size() == (32, 1, 40, 40)


def test_cbam():
    with torch.no_grad():
        model = CBAM(
            in_ch=64,
            r=16,
        )
        image = torch.empty(32, 64, 40, 40)
        out = model(image)
        assert out.size() == (32, 64, 40, 40)
