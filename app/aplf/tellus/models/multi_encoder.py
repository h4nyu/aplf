from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk, tail, take
import torch.nn as nn
from aplf.blocks import SEBlock, ResBlock, SCSE, UpSample, DownSample
import torch
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self,
                 in_ch,
                 feature_size=64,
                 depth=2,
                 r=2
                 ):
        super().__init__()
        self.down_layers = nn.ModuleList(
            [
                DownSample(
                    in_ch=in_ch,
                    out_ch=feature_size,
                ),
                *pipe(
                    range(depth),
                    map(lambda d: DownSample(
                        in_ch=int(feature_size*r**(d)),
                        out_ch=int(feature_size*r**(d + 1)),
                    )),
                    list,
                )
            ]
        )
        self.out_ch = feature_size * r ** (depth)

    def forward(self, x):
        d_outs = []
        for layer in self.down_layers:
            x, d_out = layer(x)
        return x


class LandsatEnc(nn.Module):

    def __init__(self,
                 in_ch,
                 feature_size=64,
                 depth=3,
                 ):
        super().__init__()

        self.enc = Encoder(
            in_ch=in_ch,
            feature_size=feature_size,
            depth=depth,
        )

        self.out = nn.Sequential(
            nn.Conv2d(
                in_channels=self.enc.out_ch,
                out_channels=6,
                kernel_size=1,
            ),
        )
        self.out_ch = 6
        self.before_out_ch = self.enc.out_ch

    def forward(self, x):
        before_out = self.enc(x)
        out = self.out(before_out)
        return out, before_out


class FusionEnc(nn.Module):

    def __init__(self,
                 in_ch,
                 feature_size=64,
                 depth=3,
                 ):
        super().__init__()

        self.enc = Encoder(
            in_ch=in_ch,
            feature_size=feature_size,
            depth=depth,
        )

        self.out = nn.Sequential(
            nn.Conv2d(
                in_channels=self.enc.out_ch,
                out_channels=2,
                kernel_size=1,
            ),
            nn.AdaptiveAvgPool2d(1)
        )

        self.out_ch = 2
        self.before_out_ch = self.enc.out_ch

    def forward(self, x):
        before_out = self.enc(x)
        out = self.out(before_out).view(-1, 2)
        return out, before_out


class MultiEncoder(nn.Module):
    def __init__(self,
                 feature_size,
                 resize,
                 depth,
                 ):
        super().__init__()
        self.resize = resize

        self.landsat_enc = LandsatEnc(
            in_ch=2,
            feature_size=feature_size,
            depth=depth,
        )
        self.fusion_enc = FusionEnc(
            in_ch=self.landsat_enc.before_out_ch + 2,
            feature_size=self.landsat_enc.before_out_ch,
            depth=depth,
        )
        self.pad = nn.ReplicationPad2d(4)

    def forward(self, x, part=None):
        x = self.pad(x)
        palser = F.interpolate(
            x,
            mode='bilinear',
            size=(self.resize, self.resize)
        )
        landsat, before_landsat = self.landsat_enc(palser)

        if part == 'landsat':
            return landsat

        x = pipe(
            [before_landsat, palser],
            map(lambda x: F.interpolate(x, mode='bilinear',
                                        size=(self.resize, self.resize))),
            list,
            lambda x: torch.cat(x, dim=1)
        )
        x, _ = self.fusion_enc(x)
        return x
