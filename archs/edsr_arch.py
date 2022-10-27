import torch.nn as nn
from archs import upsampler
from litsr.utils.registry import ArchRegistry


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class BasicBlock(nn.Sequential):
    def __init__(
        self,
        conv,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=False,
        bn=True,
        act=nn.ReLU(True),
    ):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
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


@ArchRegistry.register()
class EDSR_MS(nn.Module):
    def __init__(
        self,
        n_resblocks,
        n_feats,
        in_channels,
        out_channels,
        res_scale,
        which_uplayer,
        uplayer_ksize,
        interpolate_mode,
        levels,
    ):
        super().__init__()

        conv = default_conv

        kernel_size = 3
        act = nn.ReLU(True)

        # define head module
        m_head = [conv(in_channels, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
            for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        UpLayer = getattr(upsampler, which_uplayer)
        self.tail = UpLayer(
            n_feats,
            uplayer_ksize,
            out_channels,
            interpolate_mode,
            levels,
        )

    def forward(self, x, out_size):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res, out_size)

        return x
