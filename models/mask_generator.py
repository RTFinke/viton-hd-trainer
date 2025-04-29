#!/usr/bin/env python3
"""
Mask Generator for VITON-HD
Generates masks for warped clothing
"""

import torch
import torch.nn as nn
from .networks import ConvBlock, ResidualBlock, DownsampleBlock, UpsampleBlock

class MaskGenerator(nn.Module):
    def __init__(self, input_channels=23, ngf=64):
        super(MaskGenerator, self).__init__()
        
        # Input: warped cloth (3 channels) + person parse (20 channels)
        self.enc1 = ConvBlock(input_channels, ngf)
        self.enc2 = DownsampleBlock(ngf, ngf*2)
        self.enc3 = DownsampleBlock(ngf*2, ngf*4)
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(ngf*4),
            ResidualBlock(ngf*4)
        )
        
        self.dec1 = UpsampleBlock(ngf*4, ngf*2)
        self.dec2 = UpsampleBlock(ngf*2, ngf)
        
        self.out_conv = nn.Conv2d(ngf, 1, kernel_size=3, padding=1)
        self.out_activation = nn.Sigmoid()
    
    def forward(self, warped_cloth, person_parse):
        x = torch.cat([warped_cloth, person_parse], dim=1)
        
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        
        x_res = self.res_blocks(x3)
        
        x = self.dec1(x_res)
        x = self.dec2(x)
        
        mask = self.out_conv(x)
        mask = self.out_activation(mask)
        
        return {'mask': mask}