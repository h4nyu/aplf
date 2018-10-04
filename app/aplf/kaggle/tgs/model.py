from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk, tail, take
import torch.nn as nn
import torch
import torch.nn.functional as F


class CSE(nn.Module):
    def __init__(self, in_ch, r=1):
        super(CSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_ch, int(in_ch * r)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_ch * r), in_ch),
            nn.Sigmoid()
        )


    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SSE(nn.Module):
    def __init__(self, in_ch):
        super(SSE, self).__init__()

        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x

        x = self.conv(x)
        x = F.sigmoid(x)

        x = input_x * x

        return x


class SCSE(nn.Module):
    def __init__(self, in_ch, r=2 / 3):
        super(SCSE, self).__init__()

        self.cSE = CSE(in_ch, r)
        self.sSE = SSE(in_ch)

    def forward(self, x):
        cSE = self.cSE(x)
        sSE = self.sSE(x)
        x = cSE + sSE
        return x


class ResBlock(nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 ):
        super().__init__()
        if in_ch == out_ch:
            self.projection = None
        else:
            self.projection = nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=1,
            )
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_ch),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        if self.projection:
            residual = self.projection(residual)
        out += residual
        out = self.activation(out)
        return out



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
            ResBlock(
                in_ch=out_ch,
                out_ch=out_ch,
            ),
            SCSE(out_ch),
            ResBlock(
                in_ch=out_ch,
                out_ch=out_ch,
            ),
            SCSE(out_ch),
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = self.block(x)
        conv = out
        down = self.pool(conv)
        return down, conv


class UpSample(nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size=3,
                 padding=1,
                 ):
        super().__init__()
        self.block = nn.Sequential(
            ResBlock(
                in_ch,
                out_ch,
            ),
            SCSE(out_ch),
        )

    def forward(self, x, others):
        up_size = others[-1].size()[2:]
        out = pipe(
            [x, *others],
            map(lambda x: F.interpolate(x, mode='bilinear', size=up_size)),
            list
        )
        out = torch.cat([*out], 1)
        out = self.block(out)
        return out


class UNet(nn.Module):
    def __init__(self, feature_size=8, depth=4):
        super().__init__()
        self.down_layers = nn.ModuleList([
            DownSample(1, feature_size),
            *pipe(
                range(depth),
                map(lambda x: DownSample(
                    feature_size * (2 ** x),
                    feature_size * (2 ** (x + 1)),
                )),
                list,
            )
        ])

        self.center = DownSample(
            feature_size * 2 ** depth,
            feature_size * 2 ** depth,
        )

        self.up_layers = nn.ModuleList([
            *pipe(
                range(depth),
                reversed,
                map(lambda x: UpSample(
                    feature_size * (2 ** (x + 1)),
                    feature_size * (2 ** x),
                    feature_size * (2 ** (x + 1)),
                )),
                list,
            ), UpSample(
                feature_size,
                feature_size,
                feature_size,
            ),
        ])
        self._output = nn.Conv2d(
            feature_size,
            2,
            kernel_size=3
        )

    def forward(self, x):
        # down samples
        x = F.interpolate(x, mode='bilinear', size=(128, 128))
        d_outs = []
        for layer in self.down_layers:
            x, d_out = layer(x)
            d_outs.append(d_out)

        _, x = self.center(x)

        # up samples
        for layer, d_out in zip(self.up_layers, reversed(d_outs)):
            x = layer(x, d_out)

        x = self._output(x)
        x = F.interpolate(x, mode='bilinear', size=(101, 101))
        return x


class RUNet(UNet):
    def __init__(self,
                 feature_size=8,
                 depth=3,
                 ):
        super().__init__()
        self.down_layers = nn.ModuleList([
            DownSample(1, feature_size * 2 ** depth),
            *pipe(
                range(depth),
                reversed,
                map(lambda x: DownSample(
                    feature_size * (2 ** (x + 1)),
                    feature_size * (2 ** x),
                )),
                list,
            )
        ])

        self.center = DownSample(
            in_ch=feature_size,
            out_ch=feature_size,
        )

        self.up_layers = nn.ModuleList([
            *pipe(
                range(depth + 1),
                map(lambda x: UpSample(
                    feature_size * (2 ** x),
                    feature_size * (2 ** (x + 1)),
                    feature_size * (2 ** x),
                )),
                list,
            ),
        ])

        self._output = nn.Conv2d(
            feature_size * 2 ** (depth + 1),
            2,
            kernel_size=3
        )



class DUNet(UNet):
    def __init__(self,
                 feature_size=8,
                 depth=3,
                 ):
        super().__init__()
        self.down_layers = nn.ModuleList([
            DownSample(1, feature_size),
            *pipe(
                range(depth),
                map(lambda x: DownSample(
                    feature_size * (2 ** x),
                    feature_size * (2 ** (x + 1)),
                )),
                list,
            )
        ])

        self.center = DownSample(
            in_ch=feature_size * 2 ** depth,
            out_ch=feature_size * 2 ** depth,
        )


        self.up_layers = nn.ModuleList([
            *pipe(
                self.down_layers,
                reversed,
                map(lambda x: x.out_ch),
                take(depth),
                map(lambda x: UpSample(
                    feature_size * (2 ** depth),
                    feature_size * (2 ** depth),
                    x,
                )),
                list,
            ),
            UpSample(
                feature_size * 2 ** depth,
                feature_size * 2 ** depth,
                feature_size,
            ),
        ])

        self._output = nn.Conv2d(
            feature_size * 2 ** depth,
            2,
            kernel_size=3
        )



class EUNet(UNet):
    def __init__(self,
                 feature_size=8,
                 depth=3,
                 ):
        super().__init__()
        self.down_layers = nn.ModuleList([
            DownSample(1, feature_size * 2 ** depth),
            *pipe(
                range(depth),
                reversed,
                map(lambda x: DownSample(
                    feature_size * (2 ** (x + 1)),
                    feature_size * (2 ** x),
                )),
                list,
            )
        ])

        self.center = DownSample(
            in_ch=feature_size,
            out_ch=feature_size,
        )

        self.up_layers = nn.ModuleList([
            *pipe(
                self.down_layers,
                reversed,
                map(lambda x: x.out_ch),
                take(depth),
                map(lambda x: UpSample(
                    feature_size,
                    feature_size,
                    x,
                )),
                list,
            ),
            UpSample(
                feature_size,
                feature_size,
                feature_size * 2 ** depth,
            ),
        ])

        self._output = nn.Conv2d(
            feature_size,
            2,
            kernel_size=3
        )




class HUNet(UNet):
    def __init__(self, feature_size=8, depth=3):
        super().__init__()
        self.depth = depth
        self.down_layers = nn.ModuleList([
            DownSample(1, feature_size),
            *pipe(
                range(depth),
                map(lambda x: DownSample(
                    feature_size * (2 ** x),
                    feature_size * (2 ** (x + 1)),
                )),
                list,
            )
        ])

        self.center = DownSample(
            in_ch=feature_size * 2 ** depth,
            out_ch=feature_size * 2 ** depth,
        )

        self.center = DownSample(
            in_ch=feature_size * 2 ** depth,
            out_ch=feature_size * 2 ** depth,
        )

        self._cetner_output = nn.Conv2d(
            feature_size * 2 ** depth,
            2,
            kernel_size=3
        )

        down_outs = pipe(
            self.down_layers,
            map(lambda x: x.out_ch),
            list
        )
        down_outs = list(reversed(down_outs))

        up_outs = down_outs[1:]
        up_ins = pipe(
            range(depth),
            map(lambda x: [down_outs[:x+2][-1], *down_outs[:x+2]]),
            map(sum),
            list
        )

        self.up_layers = nn.ModuleList(
            pipe(
                zip(up_ins, up_outs),
                map(lambda x: UpSample(
                    in_ch=x[0],
                    out_ch=x[1]
                )),
                list,
            )
        )
        self._output = nn.Conv2d(
            up_outs[-1],
            2,
            kernel_size=3
        )

    def forward(self, x):
        # down samples
        x = F.interpolate(x, mode='bilinear', size=(128, 128))
        d_outs = []
        for layer in self.down_layers:
            x, d_out = layer(x)
            d_outs.append(d_out)

        _, x = self.center(x)
        center = self._cetner_output(x)
        d_outs = list(reversed(d_outs))[:self.depth]

        # up samples
        u_outs = []
        for i, layer in enumerate(self.up_layers):
            x = layer(x, d_outs[:i+1])

        x = self._output(x)
        x = F.interpolate(x, mode='bilinear', size=(101, 101))
        return x, center
