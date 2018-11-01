from aplf.blocks import ChannelAttention, SpatialAttention, CBAM, SEBlock, ResBlock
import torch.nn as nn
import torch
import pytest



def test_se():
    model = SEBlock(
        in_ch=64,
        out_ch=32,
        r=2,
    ).train()
    criterion = nn.MSELoss()
    image = torch.randn(32, 64, 40, 40)
    ansewer = torch.randn(32, 32, 1, 1)
    out = model(image)
    assert out.size() == ansewer.size()
    loss = criterion(out, ansewer)
    loss.backward()

def test_ca():
    model = ChannelAttention(
        in_ch=64,
        out_ch=32,
        r=2,
    ).train()
    image = torch.randn(32, 64, 40, 40)
    ansewer = torch.randn(32, 32, 1, 1)
    out = model(image)
    assert out.size() == ansewer.size()
    loss = nn.MSELoss()(out, ansewer)
    loss.backward()

def test_sa():
    model = SpatialAttention().train()
    image = torch.empty(32, 64, 40, 40)
    ansewer = torch.empty(32, 1, 40, 40)
    out = model(image)
    assert out.size() == ansewer.size()
    loss = nn.MSELoss(size_average=True)(out, ansewer)
    loss.backward()


def test_cbam():
    model = CBAM(
        in_ch=64,
        r=16,
    ).train()
    image = torch.empty(32, 64, 40, 40)
    ansewer = torch.empty(32, 64, 40, 40)
    out = model(image)
    assert out.size() == ansewer.size()
    loss = nn.MSELoss(size_average=True)(out, ansewer)
    loss.backward()

def test_res():
    model = ResBlock(
        in_ch=64,
        out_ch=32,
        r=16,
    ).train()
    image = torch.empty(32, 64, 40, 40)
    ansewer = torch.empty(32, 32, 40, 40)
    out = model(image)
    assert out.size() == ansewer.size()
    loss = nn.MSELoss(size_average=True)(out, ansewer)
    loss.backward()
