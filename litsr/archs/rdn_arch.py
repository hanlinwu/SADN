# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from . import common

import torch
import torch.nn as nn
from litsr.utils.registry import ArchRegistry


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(
            *[nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1), nn.ReLU()]
        )

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


@ArchRegistry.register()
class RDN(nn.Module):
    def __init__(
        self,
        scale,
        num_features,
        num_blocks,
        num_layers,
        rgb_range,
        in_channels,
        out_channels,
        rgb_mean=(0.4488, 0.4371, 0.4040),
        rgb_std=(1.0, 1.0, 1.0),
    ):
        super().__init__()
        r = scale
        G0 = num_features
        kSize = 3

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = [num_blocks, num_layers, num_features]
        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(
            in_channels, G0, kSize, padding=(kSize - 1) // 2, stride=1
        )
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(RDB(growRate0=G0, growRate=G, nConvLayers=C))

        # Global Feature Fusion
        self.GFF = nn.Sequential(
            *[
                nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
                nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1),
            ]
        )

        # Up-sampling net
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(
                *[
                    nn.Conv2d(G0, G * r * r, kSize, padding=(kSize - 1) // 2, stride=1),
                    nn.PixelShuffle(r),
                    nn.Conv2d(
                        G, out_channels, kSize, padding=(kSize - 1) // 2, stride=1
                    ),
                ]
            )
        elif r == 4:
            self.UPNet = nn.Sequential(
                *[
                    nn.Conv2d(G0, G * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, G * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(
                        G, out_channels, kSize, padding=(kSize - 1) // 2, stride=1
                    ),
                ]
            )

    def forward(self, x, return_features=False):
        x = self.sub_mean(x)
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        feat = x + f__1

        out = self.UPNet(feat)
        out = self.add_mean(out)

        if return_features:
            return out, feat
        return out


@ArchRegistry.register()
class RDN_MS_Factory(RDN):
    """
    The multi scale version of RDN,
    and you can specify rgb_mean/rgb_std/rgb_range!
    """

    def __init__(
        self,
        UPNet,
        num_features,
        num_blocks,
        num_layers,
        rgb_range,
        in_channels,
        out_channels,
        rgb_mean=(0.4488, 0.4371, 0.4040),
        rgb_std=(1.0, 1.0, 1.0),
    ):
        super().__init__(
            scale=0,
            num_features=num_features,
            num_blocks=num_blocks,
            num_layers=num_layers,
            rgb_range=rgb_range,
            in_channels=in_channels,
            out_channels=out_channels,
            rgb_mean=rgb_mean,
            rgb_std=rgb_std,
        )
        # Redefine up-sampling net
        self.UPNet = UPNet

    def forward(self, x, out_size):
        x = self.sub_mean(x)
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        x = self.UPNet(x, out_size)
        x = self.add_mean(x)
        return x
