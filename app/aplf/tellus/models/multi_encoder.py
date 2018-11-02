from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk, tail, take
import torch.nn as nn
from aplf.blocks import SEBlock, ResBlock, SCSE, UpSample, DownSample, ChannelAttention 
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


class MultiEncoder(nn.Module):
    def __init__(self,
                 feature_size=64,
                 resize=120,
                 depth=3,
                 pad=4,
                 ):
        super().__init__()
        self.resize = resize

        self.landsat_enc = Encoder(
            in_ch=2,
            feature_size=feature_size,
            depth=1,
        )
        self.landsat_out = nn.Sequential(
            nn.Conv2d(
                in_channels=self.landsat_enc.out_ch,
                out_channels=6,
                kernel_size=3,
            ),
            nn.Upsample(size=(4, 4), mode='bilinear'),
            nn.Sigmoid(),
        )

        self.fusion_enc = Encoder(
            in_ch=8,
            feature_size=feature_size,
            depth=depth,
        )

        self.logit_out = nn.Sequential(
            nn.Conv2d(
                in_channels=self.fusion_enc.out_ch,
                out_channels=2,
                kernel_size=1,
            ),
            nn.AdaptiveAvgPool2d(1)
        )

        self.pad = nn.ReflectionPad2d(pad)

    def forward(self, x):
        x = F.interpolate(
            x,
            mode='bilinear',
            size=(self.resize, self.resize)
        )
        x = self.pad(x)
        palser_x = self.pad(x)
        landsat_x = self.landsat_enc(x)
        landsat_x = self.landsat_out(landsat_x)

        x = pipe(
            [landsat_x, palser_x],
            map(lambda x: F.interpolate(x, mode='bilinear',
                                        size=(self.resize, self.resize))),
            list,
            lambda x: torch.cat(x, dim=1)
        )
        x = self.fusion_enc(x)
        x = self.logit_out(x).view(-1, 2)
        return x, landsat_x
