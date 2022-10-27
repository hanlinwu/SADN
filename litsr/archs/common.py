import math

import torch
import torch.nn as nn
from litsr.utils.logger import logger
from torch.nn.utils import weight_norm

################
# Basic blocks
################


def spatial_fold(input, fold):
    if fold == 1:
        return input

    batch, channel, height, width = input.shape
    h_fold = height // fold
    w_fold = width // fold

    return (
        input.view(batch, channel, h_fold, fold, w_fold, fold)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(batch, -1, h_fold, w_fold)
    )


def spatial_unfold(input, unfold):
    if unfold == 1:
        return input

    batch, channel, height, width = input.shape
    h_unfold = height * unfold
    w_unfold = width * unfold

    return (
        input.view(batch, -1, unfold, unfold, height, width)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(batch, -1, h_unfold, w_unfold)
    )


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    logger.warning("The module is deprecated, and will be removed in the future! ")
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class WeightNormedConv(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        act=nn.ReLU(True),
    ):
        conv = weight_norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                stride=stride,
                bias=bias,
            )
        )
        m = [conv]
        if act:
            m.append(act)
        super().__init__(*m)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        if len(rgb_std) != len(rgb_mean):
            assert len(rgb_std) == 1
            rgb_std = rgb_std * len(rgb_mean)
        channel = len(rgb_mean)
        super(MeanShift, self).__init__(channel, channel, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(channel).view(channel, channel, 1, 1)
        self.weight.data.div_(std.view(channel, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=False,
        bn=True,
        act=nn.ReLU(True),
    ):

        m = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=(kernel_size // 2),
                stride=stride,
                bias=bias,
            )
        ]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
        self,
        conv,
        n_feats,
        kernel_size,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


def make_coord(shape, ranges=None, flatten=True):
    """Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret
