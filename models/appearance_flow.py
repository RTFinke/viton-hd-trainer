#!/usr/bin/env python3
"""
Appearance Flow Network for VITON-HD
Based on the paper: Diffusion VTON: High-Fidelity Virtual Try-On Network via Mask-Aware Diffusion Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import ConvBlock, ResidualBlock, DownsampleBlock, UpsampleBlock

class AppearanceFlowNet(nn.Module):
    """
    Appearance Flow Network for warping clothes to match the pose of the person
    """
    def __init__(self, input_channels_person=3, input_channels_cloth=3, input_channels_parse=20, ngf=64):
        super(AppearanceFlowNet, self).__init__()
        
        total_input_channels = input_channels_person + input_channels_cloth + input_channels_parse
        
        # Encoder
        self.enc1 = ConvBlock(total_input_channels, ngf)
        self.enc2 = DownsampleBlock(ngf, ngf*2)
        self.enc3 = DownsampleBlock(ngf*2, ngf*4)
        self.enc4 = DownsampleBlock(ngf*4, ngf*8)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(ngf*8),
            ResidualBlock(ngf*8),
            ResidualBlock(ngf*8)
        )
        
        # Decoder
        self.dec1 = UpsampleBlock(ngf*8, ngf*4)
        self.dec2 = UpsampleBlock(ngf*4, ngf*2)
        self.dec3 = UpsampleBlock(ngf*2, ngf)
        
        # Output layer
        self.out_conv = nn.Conv2d(ngf, 3, kernel_size=3, padding=1)
        self.out_activation = nn.Tanh()
    
    def forward(self, person_image, cloth_image, person_parse):
        x = torch.cat([person_image, cloth_image, person_parse], dim=1)
        
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        
        # Residual
        x_res = self.res_blocks(x4)
        
        # Decoder
        x = self.dec1(x_res)
        x = self.dec2(x)
        x = self.dec3(x)
        
        # Output
        warped_cloth = self.out_conv(x)
        warped_cloth = self.out_activation(warped_cloth)
        
        return {'warped_cloth': warped_cloth}
