from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk, tail, take
import torch.nn as nn
from aplf.blocks import SEBlock, ResBlock, SCSE, UpSample
import torch
import torch.nn.functional as F


class DownSample(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size=3,
                 padding=1,
                 ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.block = nn.Sequential(
            ResBlock(
                in_ch=in_ch,
                out_ch=out_ch,
            ),
            SCSE(out_ch),
        )
        self.block = nn.Sequential(
            ResBlock(
                in_ch=in_ch,
                out_ch=out_ch,
            ),
            SCSE(out_ch),
        )

        self.down = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x):
        out = self.block(x)
        conv = out
        down = self.down(conv)
        return down, conv


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


class UNet(nn.Module):
    def __init__(self,
                 in_ch,
                 feature_size=64,
                 depth=3,
                 ratio=2):
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
                        in_ch=int(feature_size*ratio**(d)),
                        out_ch=int(feature_size*ratio**(d + 1)),
                    )),
                    list,
                )
            ]
        )
        self.center = DownSample(
            in_ch=feature_size*ratio**depth,
            out_ch=feature_size*ratio**depth,
        )
        self.up_layers = nn.ModuleList([
            *pipe(
                range(depth),
                reversed,
                map(lambda l: UpSample(
                    in_ch=feature_size *
                    ratio**(l+1) + feature_size*ratio**(l+1),
                    out_ch=feature_size*ratio**l,
                )),
                list,
            ),
            UpSample(
                in_ch=feature_size + feature_size,
                out_ch=feature_size,
            ),
        ])
        self.out_ch = feature_size

    def forward(self, x):
        d_outs = []
        for layer in self.down_layers:
            x, d_out = layer(x)
            d_outs.append(d_out)
        d_outs = list(reversed(d_outs))
        x, _ = self.center(x)
        u_outs = []
        for d, layer in zip(d_outs, self.up_layers):
            x = layer(x, [d])
        return x


class MultiEncoder(nn.Module):
    def __init__(self,
                 feature_size=64,
                 resize=120,
                 depth=2,
                 pad=4,
                 ):
        super().__init__()
        self.resize = resize

        self.denoise_enc = UNet(
            in_ch=1,
            feature_size=feature_size,
        )
        self.palsar_out = nn.Sequential(
            nn.Conv2d(
                in_channels=self.denoise_enc.out_ch,
                out_channels=1,
                kernel_size=3,
            ),
            nn.Upsample(size=(40, 40), mode="bilinear")
        )

        self.landsat_enc = UNet(
            in_ch=1,
            feature_size=feature_size,
            depth=depth,
        )
        self.landsat_out = nn.Sequential(
            nn.Conv2d(
                in_channels=self.landsat_enc.out_ch,
                out_channels=3,
                kernel_size=3,
            ),
            nn.Upsample(size=(4, 4), mode='bilinear')
        )
        self.fusion_enc = Encoder(
            feature_size=8,
            in_ch=8,
            depth=depth,
        )
        self.logit_out = nn.Sequential(
            nn.Conv2d(
                in_channels=self.fusion_enc.out_ch,
                out_channels=2,
                kernel_size=3,
            ),
            nn.AdaptiveAvgPool2d(1)
        )

        self.pad = nn.ZeroPad2d(pad)

    def forward(self, b_x, a_x):
        b_x = F.interpolate(b_x, mode='bilinear',
                            size=(self.resize, self.resize))
        a_x = F.interpolate(a_x, mode='bilinear',
                            size=(self.resize, self.resize))
        b_x = self.pad(b_x)
        a_x = self.pad(a_x)

        p_before = self.denoise_enc(b_x)
        p_after = self.denoise_enc(a_x)
        p_before = self.palsar_out(p_before)
        p_after = self.palsar_out(p_after)
        #
        l_before = self.landsat_enc(b_x)
        l_after = self.landsat_enc(a_x)
        l_before = self.landsat_out(l_before)
        l_after = self.landsat_out(l_after)
        x = pipe(
            [l_before, l_after, p_before, p_after],
            map(lambda x: F.interpolate(x, mode='bilinear',
                                        size=(self.resize, self.resize))),
            list,
            lambda x: torch.cat(x, dim=1)
        )
        print(x.size())
        x = self.fusion_enc(x)
        print(x.size())
        x = self.logit_out(x).view(-1, 2)
        return x, p_before, p_after, l_before, l_after


class AE(nn.Module):
    def __init__(self,
                 in_size=(2, 40, 40),
                 out_size=(2, 40, 40),
                 center_out_size=(6, 4, 4),
                 feature_size=64,
                 resize=120,
                 pad=4,
                 ):
        super().__init__()
        self.resize = resize
        self.in_size = in_size
        self.out_size = out_size
        self.center_out_size = center_out_size
        self.down_layers = nn.ModuleList([
            DownSample(in_size[0], feature_size),
            DownSample(feature_size, feature_size * 2 ** 1),
            DownSample(feature_size * 2 ** 1, feature_size * 2 ** 2),
            DownSample(feature_size * 2 ** 2, feature_size * 2 ** 3),
        ])

        self.center = DownSample(
            in_ch=feature_size * 2 ** 3,
            out_ch=feature_size * 2 ** 3,
        )

        self.center_out = nn.Conv2d(
            feature_size * 2 ** 3,
            center_out_size[0],
            kernel_size=3
        )

        self.up_layers = nn.ModuleList([
            UpSample(
                in_ch=feature_size * 16,
                out_ch=feature_size,
            ),
            UpSample(
                in_ch=feature_size * 13,
                out_ch=feature_size,
            ),
            UpSample(
                in_ch=feature_size * 7,
                out_ch=feature_size
            ),
            UpSample(
                in_ch=feature_size * 4,
                out_ch=feature_size
            ),
        ])
        self._output = nn.Conv2d(
            feature_size + center_out_size[0],
            out_size[0],
            kernel_size=3
        )
        self.pad = nn.ZeroPad2d(pad)

    def forward(self, x):
        x = F.interpolate(
            x,
            mode='bilinear',
            size=(self.resize, self.resize)
        )
        x = self.pad(x)

        d_outs = []
        for layer in self.down_layers:
            x, d_out = layer(x)
            d_outs.append(d_out)

        _, x = self.center(x)
        center = self.center_out(x)
        center = F.interpolate(
            center,
            mode='bilinear',
            size=(self.center_out_size[1], self.center_out_size[2])
        )

        d_outs = list(reversed(d_outs))

        # up samples
        u_outs = []
        for i, layer in enumerate(self.up_layers):
            x = layer(x, d_outs[:i+1][-2:])

        x = torch.cat(
            [
                x,
                F.interpolate(
                    center,
                    size=x.size()[2:],
                    mode='bilinear',
                )
            ],
            dim=1
        )
        x = self._output(x)
        x = F.interpolate(
            x,
            mode='bilinear',
            size=(self.out_size[1], self.out_size[2]),
        )
        return x, center
