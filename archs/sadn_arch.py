import torch
import torch.nn as nn
from litsr.archs import common
from litsr.data import *
from litsr.utils.registry import ArchRegistry
from matplotlib import pyplot as plt
from torch.nn.utils import weight_norm
import time
from os.path import exists
import os

from archs import upsampler

from .dynamic_layers import ScaleAwareDynamicConv2d


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class WideConvBlock(nn.Module):
    def __init__(self, num_features, kernel_size, width_multiplier=4, reduction=4):
        super().__init__()

        self.body = nn.Sequential(
            *[
                common.WeightNormedConv(
                    num_features, int(num_features * width_multiplier), 3
                ),
                common.WeightNormedConv(
                    int(num_features * width_multiplier), num_features, 3, act=None
                ),
                common.WeightNormedConv(
                    num_features,
                    num_features,
                    kernel_size,
                    act=None,
                    # res_scale=res_scale,
                ),
                SEBlock(num_features, reduction),
            ]
        )

    def forward(self, x, scale):
        return x + self.body(x)


class DynamicWideConvBlock(nn.Module):
    def __init__(
        self,
        num_features,
        kernel_size,
        width_multiplier=4,
        dynamic_K=4,
        reduction=4,
    ):
        super().__init__()

        self.body = nn.Sequential(
            *[
                common.WeightNormedConv(
                    num_features,
                    int(num_features * width_multiplier),
                    kernel_size,
                    # res_scale=2.0,
                ),
                common.WeightNormedConv(
                    int(num_features * width_multiplier),
                    num_features,
                    kernel_size,
                    act=None,
                ),
            ]
        )
        self.d_conv = weight_norm(
            ScaleAwareDynamicConv2d(
                num_features,
                num_features,
                kernel_size,
                padding=kernel_size // 2,
                K=dynamic_K,
            )
        )
        self.se_block = SEBlock(num_features, reduction)

    def forward(self, x, scale):
        r = self.body(x)
        r = self.d_conv(r, scale)
        r = self.se_block(r)
        return x + r


class LocalDenseGroup(nn.Module):
    def __init__(
        self,
        num_features,
        width_multiplier,
        num_layers,
        reduction,
        use_dynamic_conv,
        dynamic_K,
    ):
        super().__init__()
        kSize = 3
        self.num_layers = num_layers

        self.ConvBlockList = nn.ModuleList()
        self.compressList = nn.ModuleList()
        self.use_dynamic_conv = use_dynamic_conv
        for idx in range(num_layers):
            if use_dynamic_conv:
                self.ConvBlockList.append(
                    DynamicWideConvBlock(
                        num_features,
                        kSize,
                        width_multiplier=width_multiplier,
                        # res_scale=1 / math.sqrt(num_layers),
                        dynamic_K=dynamic_K,
                        reduction=reduction,
                    )
                )
            else:
                self.ConvBlockList.append(
                    WideConvBlock(
                        num_features,
                        kSize,
                        width_multiplier=width_multiplier,
                        # res_scale=1 / math.sqrt(num_layers),
                        reduction=reduction,
                    )
                )
        for idx in range(1, num_layers):
            self.compressList.append(
                common.WeightNormedConv(
                    (idx + 1) * num_features, num_features, 1, act=None
                )
            )

    def forward(self, x, scale):
        concat = x
        for l in range(self.num_layers):
            if l == 0:
                out = self.ConvBlockList[l](concat, scale)
            else:
                concat = torch.cat([concat, out], dim=1)
                out = self.compressList[l - 1](concat)
                out = self.ConvBlockList[l](out, scale)
        return out


class FeedbackBlock(nn.Module):
    def __init__(
        self,
        num_features,
        width_multiplier,
        num_layers,
        num_groups,
        reduction,
        use_dynamic_conv,
        dynamic_K,
    ):
        super().__init__()
        kSize = 3
        self.num_groups = num_groups

        self.LDGList = nn.ModuleList()
        for _ in range(num_groups):
            self.LDGList.append(
                LocalDenseGroup(
                    num_features,
                    width_multiplier,
                    num_layers,
                    reduction,
                    use_dynamic_conv,
                    dynamic_K,
                )
            )

        self.compressList = nn.ModuleList()
        for idx in range(1, num_groups):
            self.compressList.append(
                common.WeightNormedConv(
                    (idx + 1) * num_features, num_features, 1, act=None
                )
            )

        self.compress_in = common.WeightNormedConv(
            2 * num_features, num_features, kSize
        )

        self.should_reset = True
        self.last_hidden = None

    def forward(self, x, scale):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size(), device=x.device)
            self.last_hidden.copy_(x)
            self.should_reset = False

        x = torch.cat((x, self.last_hidden), 1)

        concat = self.compress_in(x)
        for l in range(self.num_groups):
            if l == 0:
                out = self.LDGList[l](concat, scale)
            else:
                concat = torch.cat([concat, out], dim=1)
                out = self.compressList[l - 1](concat)
                out = self.LDGList[l](out, scale)

        self.last_hidden = out
        return out

    def reset_state(self):
        self.should_reset = True


@ArchRegistry.register()
class SADN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_features,
        num_layers,
        num_groups,
        reduction,
        width_multiplier,
        interpolate_mode,
        levels,
        use_dynamic_conv,
        dynamic_K,
        which_uplayer,
        uplayer_ksize,
        rgb_range,
        rgb_mean,
        rgb_std,
    ):
        super().__init__()
        kernel_size = 3
        skip_kernel_size = 5
        num_inputs = in_channels
        n_feats = num_features
        self.interpolate_mode = interpolate_mode
        self.levels = levels

        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(
            *[common.WeightNormedConv(num_inputs, num_features, kernel_size)]
        )

        self.body = FeedbackBlock(
            num_features,
            width_multiplier,
            num_layers,
            num_groups,
            reduction,
            use_dynamic_conv,
            dynamic_K,
        )

        self.tail = nn.Sequential(
            *[
                common.WeightNormedConv(
                    num_features, num_features, kernel_size, act=None
                )
            ]
        )

        self.skip = common.WeightNormedConv(
            num_inputs, num_features, skip_kernel_size, act=None
        )

        UpLayer = getattr(upsampler, which_uplayer)
        self.uplayer = UpLayer(
            n_feats,
            uplayer_ksize,
            out_channels,
            interpolate_mode,
            levels,
        )

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, ScaleAwareDynamicConv2d):
                m.update_temperature()

    def forward(self, x, out_size):

        self.body.reset_state()
        if isinstance(out_size, int):
            out_size = [out_size, out_size]
        scale = torch.tensor([x.shape[2] / out_size[0]], device=x.device)
        x = self.sub_mean(x)
        skip = self.skip(x)

        x = self.head(x)
        h_list = []

        for _ in range(self.levels):
            h = self.body(x, scale)
            h = self.tail(h)
            h = h + skip
            h_list.append(h)

        x = self.uplayer(h_list, out_size)

        x = self.add_mean(x)

        return x


@ArchRegistry.register()
class SADN_vis(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_features,
        num_layers,
        num_groups,
        reduction,
        width_multiplier,
        interpolate_mode,
        levels,
        use_dynamic_conv,
        dynamic_K,
        which_uplayer,
        uplayer_ksize,
        rgb_range,
        rgb_mean,
        rgb_std,
    ):
        super().__init__()
        kernel_size = 3
        skip_kernel_size = 5
        num_inputs = in_channels
        n_feats = num_features
        self.interpolate_mode = interpolate_mode
        self.levels = levels

        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(
            *[common.WeightNormedConv(num_inputs, num_features, kernel_size)]
        )

        self.use_dynamic_conv = use_dynamic_conv
        self.body = FeedbackBlock(
            num_features,
            width_multiplier,
            num_layers,
            num_groups,
            reduction,
            use_dynamic_conv,
            dynamic_K,
        )

        self.tail = nn.Sequential(
            *[
                common.WeightNormedConv(
                    num_features, num_features, kernel_size, act=None
                )
            ]
        )

        self.skip = common.WeightNormedConv(
            num_inputs, num_features, skip_kernel_size, act=None
        )

        UpLayer = getattr(upsampler, which_uplayer)
        self.uplayer = UpLayer(
            n_feats,
            uplayer_ksize,
            out_channels,
            interpolate_mode,
            levels,
        )

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, ScaleAwareDynamicConv2d):
                m.update_temperature()

    def forward(self, x, out_size):
        self.body.reset_state()
        if isinstance(out_size, int):
            out_size = [out_size, out_size]
        scale = torch.tensor([x.shape[2] / out_size[0]], device=x.device)
        x = self.sub_mean(x)
        skip = self.skip(x)

        x = self.head(x)
        h_list = []

        for _ in range(self.levels):
            h = self.body(x, scale)
            h = self.tail(h)
            h = h + skip
            h_list.append(h)
        vis = torch.mean(h_list[-1], dim=1)
        vis = (vis - vis.min()) / (vis.max() - vis.min())
        vis = vis[..., 88:217, 32:161]
        # vis = vis + 0.2
        # vis.clamp_max_(1)
        print(torch.min(vis), torch.max(vis))
        # print(vis.shape)

        savepath = "logs/vis"
        filename = "geo_residential_t7.png"

        if self.use_dynamic_conv:
            savepath = os.path.join(savepath, "dy" + filename.replace(".png", ""))
        else:
            savepath = os.path.join(savepath, "wo_dy" + filename.replace(".png", ""))
        if not exists(savepath):
            os.mkdir(savepath)

        savepath = os.path.join(savepath, "x{0}.png".format(int((1 / scale).item())))

        plt.imsave(savepath, vis.cpu().numpy()[0], cmap="hsv")

        x = self.uplayer(h_list, out_size)

        x = self.add_mean(x)

        return x
