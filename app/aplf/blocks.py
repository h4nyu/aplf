from cytoolz.curried import keymap, filter, pipe, merge, map, reduce, topk, tail, take
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, r=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch//r),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch//r, out_ch),
            nn.Sigmoid()
        )
        self.out_ch = out_ch

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_ch, 1, 1)
        return y


class CSE(nn.Module):
    def __init__(self, in_ch, r=1):
        super().__init__()
        self.se = SEBlock(in_ch, in_ch, r=r)

    def forward(self, x):
        y = self.se(x)
        return x * y


class SSE(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x

        x = self.conv(x)
        x = F.sigmoid(x)

        x = input_x * x

        return x


class SCSE(nn.Module):
    def __init__(self, in_ch, r=2):
        super(SCSE, self).__init__()

        self.cSE = CSE(in_ch, r)
        self.sSE = SSE(in_ch)

    def forward(self, x):
        cSE = self.cSE(x)
        sSE = self.sSE(x)
        x = cSE + sSE
        return x


class ChannelAttention(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 r=16,
                 bias=False,
                 has_activate=True):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.out_ch = out_ch
        self.fc = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//r, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch//r, out_ch, kernel_size=1, bias=bias),
        )
        self.has_activate = has_activate
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        if self.has_activate:
            return self.sigmoid(out)
        else:
            return out


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self,
                 in_ch,
                 r=16,
                 kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(
            in_ch,
            out_ch=in_ch,
            r=r,
            has_activate=True,
        )
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class ResBlock(nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 r=2,
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
        self.conv3bn = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch//2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.BatchNorm2d(out_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_ch//2,
                out_ch//2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.BatchNorm2d(out_ch // 2),
        )

        self.conv5bn = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch//2,
                kernel_size=5,
                padding=2,
                stride=1,
            ),
            nn.BatchNorm2d(out_ch//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_ch//2,
                out_ch//2,
                kernel_size=5,
                padding=2,
                stride=1,
            ),
            nn.BatchNorm2d(out_ch // 2),
        )
        self.ca = ChannelAttention(
            in_ch=(out_ch // 2)*2,
            out_ch=out_ch,
        )
        self.sa = SpatialAttention()

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out3 = self.conv3bn(x)
        out5 = self.conv5bn(x)
        out = torch.cat([out5, out3], dim=1)
        out = self.ca(out) * out
        out = self.sa(out) * out
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
                 r=2,
                 ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.block = nn.Sequential(
            ResBlock(
                in_ch=in_ch,
                out_ch=out_ch,
            ),
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
                 r=2,
                 ):
        super().__init__()
        self.block = nn.Sequential(
            ResBlock(
                in_ch,
                out_ch,
            ),
        )

    def forward(self, x, others, size):
        out = pipe(
            [x, *others],
            map(lambda x: F.interpolate(x, mode='bilinear', size=size)),
            list
        )
        out = torch.cat([*out], 1)
        out = self.block(out)
        return out
