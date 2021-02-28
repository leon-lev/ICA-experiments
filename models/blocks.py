""" 
Adapted from https://github.com/milesial/Pytorch-UNet

Parts of the U-Net model
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 mid_channels=None, downsample=False,
                 activation='ReLU', **kwargs):
        super(ResidualBlock, self).__init__()

        if mid_channels == None:
            mid_channels = out_channels

        self.activation = getattr(nn, activation)(**kwargs)

        skip_block = []
        if downsample:
            skip_block += [nn.AvgPool2d(2, 2)]
        if in_channels != out_channels:
            skip_block += [nn.Conv2d(in_channels, out_channels, 1)]

        if skip_block:
            self.skip_block = nn.Sequential(*skip_block)
        else:
            self.skip_block = None

        stride = 2 if downsample else 1

        conv_block = [  nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                        nn.InstanceNorm2d(mid_channels),
                        self.activation,
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(mid_channels, out_channels, 3, stride, bias=False),
                        nn.InstanceNorm2d(out_channels)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        x_conv = self.conv_block(x)
        if self.skip_block is not None:
            x = self.skip_block(x)
        out = x + x_conv
        out = self.activation(out)
        return out


class DoubleConv(nn.Module):
    """(convolution => [IN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, 
                 mid_channels=None, activation='ReLU', **kwargs):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.activation = getattr(nn, activation)(**kwargs)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            self.activation,
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            self.activation
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels,
                       activation='LeakyReLU',
                       negative_slope=0.2,
                       inplace=True)
        )

    def forward(self, x):
        return self.avgpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)