import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
from models.attentions import simam_module



class MWFE(nn.Module):
    def __init__(self, in_channel, r=0.5):  # channel为输入的维度, r为全连接层缩放比例->控制中间层个数
        super(MWFE, self).__init__()
        # 全局最大池化
        self.MaxPool = nn.AdaptiveMaxPool2d(1)

        self.fc_MaxPool = nn.Sequential(
            nn.Conv2d(in_channel, int(in_channel * r), kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(int(in_channel * r), in_channel, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # 全局均值池化
        self.AvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc_AvgPool = nn.Sequential(
            nn.Conv2d(in_channel, int(in_channel * r), kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(int(in_channel * r), in_channel, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )


        self.fc_Pool = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        residul = x

        max_branch = self.MaxPool(x)
        max_weight = self.fc_MaxPool(max_branch)

        avg_branch = self.AvgPool(x)
        avg_weight = self.fc_AvgPool(avg_branch)

        Local_weight = self.fc_Pool(x)

        weight = max_weight + avg_weight + Local_weight
        weight = self.sigmoid(weight)

        # x = Mc * x
        x = weight * x

        x = x + residul
        return x


class SEM(nn.Module):
    def __init__(self, nf, reduction=2, stride=1, dilation=1):
        super(SEM, self).__init__()
        group_width = nf // reduction

        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)

        self.k1 = nn.Sequential(
            nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                bias=False)
        )
        self.k2 = nn.Sequential(
            nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                bias=False)
        )
        # self.PAConv = PAConv(group_width)
        self.SIMAM = simam_module(group_width, e_lambda=1e-4)

        self.MWFE = MWFE(in_channel=group_width * reduction, r=0.5)

        self.conv3 = nn.Conv2d(
            group_width * reduction, nf, kernel_size=1, bias=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = x


        out_a = self.conv1_a(x)
        out_a = self.lrelu(out_a)
        out_a = self.k1(out_a)
        out_a = self.lrelu(out_a)

        out_b = self.conv1_b(x)
        out_b = self.lrelu(out_b)
        out_b = self.SIMAM(out_b)
        out_b = self.k2(out_b)
        out_b = self.lrelu(out_b)

        out = torch.cat([out_a, out_b], dim=1)

        out = self.MWFE(out)
        out = self.conv3(out)
        out += residual

        return out





def sequential(*args):
    """
    Modules will be added to the a Sequential Container in the order they
    are passed.

    Parameters
    ----------
    args: Definition of Modules in order.
    -------
    """
    if len(args) == 1:
        if isinstance(args[0]):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value



def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2),
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)


def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = conv_layer(in_channels,
                      out_channels * (upscale_factor ** 2),
                      kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

class DAU(nn.Module):
    def __init__(self, channels=64, r=4,out_nc=3, upscale=4):
        super(DAU, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(

            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )

        self.sigmoid = nn.Sigmoid()

        self.upsampler = pixelshuffle_block(channels, out_nc, upscale_factor=upscale)

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        xo = self.upsampler(xo)

        return xo

class ESDAN(nn.Module):

    def __init__(self, in_nc, out_nc, nf,  nb, scale=4):
        super(ESDAN, self).__init__()
        SEM_block_f = functools.partial(SEM, nf=nf, reduction=2)

        self.scale = scale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1,
                                    bias=True)  # torch.Size([32, 3, 64, 64]) -> torch.Size([32, 40, 64, 64])
        ### main blocks
        self.SEM_trunk = arch_util.make_layer(SEM_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.dau = DAU(channels=nf, r=6, out_nc=out_nc, upscale=scale)


    def forward(self, x):

        fea = self.conv_first(x)

        trunk = self.SEM_trunk(fea)
        trunk = self.trunk_conv(trunk)

        out = self.dau(fea, trunk)


        ILR = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)

        out = out + ILR
        return out
