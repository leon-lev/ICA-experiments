""" 
Adapted from https://github.com/milesial/Pytorch-UNet

Full assembly of the parts to form the complete network 
"""

import torch
import torch.nn as nn
from torch.nn.modules import activation

from .blocks import Up, Down, DoubleConv, OutConv


class UNetModel(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, n_classes,
                 output_activation='none'):

        super(UNetModel, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels_in, 64)

        # downsampling
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.down5 = Down(512, 512)
        self.down6 = Down(512, 512)
        self.down7 = Down(512, 512)
        self.down8 = Down(512, 512)

        # upsampling
        self.up1 = Up(1024, 512)
        self.up2 = Up(1024, 512)
        self.up3 = Up(1024, 512)
        self.up3 = Up(1024, 512)
        self.up4 = Up(1024, 512)
        self.up5 = Up(1024, 256)
        self.up6 = Up(512, 128)
        self.up7 = Up(256, 64)
        self.up8 = Up(128, 64)

        self.outc = OutConv(64, n_channels_out)
        
        if output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = None

        # conditional encoding block
        embedding_dims = 64
        self.embedding = nn.Embedding(n_classes, embedding_dims)
        
        cond_encoder = []
        cond_out_neurons = [256, 512]
        cond_in_neurons = embedding_dims
        for c in cond_out_neurons:
            cond_encoder += [nn.Linear(in_features=cond_in_neurons,
                                       out_features=c,
                                       bias=False),
                             nn.BatchNorm1d(c),
                             nn.ReLU(inplace=True)]
            cond_in_neurons = c
        
        self.cond_encoder = nn.Sequential(*cond_encoder)
        self.bottelneck_conv = DoubleConv(1024, 512,
                                          activation='LeakyReLU',
                                          negative_slope=0.2,
                                          inplace=True)

    def forward(self, x, cond):
        x1 = self.inc(x)
        # downsampling
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        x = self.down8(x8)
        
        # conditional encoding block
        e = self.embedding(cond)
        e = self.cond_encoder(e)
        e = e.reshape(-1, 512, 1, 1)
        # combine cond with bottelneck vector
        x = torch.cat([e, x], dim=1)
        x = self.bottelneck_conv(x)
        # upsampling
        x = self.up1(x, x8)
        x = self.up2(x, x7)
        x = self.up3(x, x6)
        x = self.up4(x, x5)
        x = self.up5(x, x4)
        x = self.up6(x, x3)
        x = self.up7(x, x2)
        x = self.up8(x, x1)
        x = self.outc(x)

        if self.output_activation is not None:
            code = self.output_activation(x)
        else:
            code = x
        return code