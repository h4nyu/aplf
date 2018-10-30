from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk, tail, take
import torch.nn as nn
from aplf.blocks import SEBlock, ResBlock, SCSE, UpSample
import torch
import torch.nn.functional as F





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

        self.center_out = nn.Sequential(
            nn.Conv2d(
                feature_size * 2 ** 3,
                center_out_size[0],
                kernel_size=3
            ),
            nn.Sigmoid()
        )

        self.up_layers = nn.ModuleList([
            UpSample(
                in_ch=feature_size * 2 ** 3,
                out_ch=feature_size * 2 ** 2,
            ),
            UpSample(
                in_ch=feature_size * 2 ** 2,
                out_ch=feature_size * 2 ** 1,
            ),
            UpSample(
                in_ch=feature_size * 2 ** 1,
                out_ch=feature_size
            ),
            UpSample(
                in_ch=feature_size,
                out_ch=feature_size
            ),
        ])
        self._output = nn.Sequential(
            nn.Conv2d(
                feature_size + center_out_size[0],
                out_size[0],
                kernel_size=3
            ),
            nn.Sigmoid()
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
        for layer in self.up_layers:
            _, _, h, w = x.size()
            x = layer(x, [], size=(2*h, 2*w))

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
