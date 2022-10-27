# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

import torch
from litsr.archs.rdn_arch import *
from litsr.utils.registry import ArchRegistry
from easydict import EasyDict

from archs import upsampler


@ArchRegistry.register()
class RDN_MS(RDN):
    """
    The multi scale version of RDN, and you can specify rgb_mean/rgb_std/rgb_range!
    """

    def __init__(self, **args):
        args = EasyDict(args)
        super().__init__(
            scale=0,
            num_features=args.num_features,
            num_blocks=args.num_blocks,
            num_layers=args.num_layers,
            rgb_range=args.rgb_range,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
        )
        # Redefine up-sampling net
        UpLayer = getattr(upsampler, args.which_uplayer)
        self.UPNet = UpLayer(
            args.num_features, 3, args.out_channels, args.interpolate_mode, args.levels
        )

        rgb_mean = args.get("rgb_mean", (0.4488, 0.4371, 0.4040))
        rgb_std = args.get("rgb_std", (1.0, 1.0, 1.0))
        rgb_range = args.get("rgb_range")
        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

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
