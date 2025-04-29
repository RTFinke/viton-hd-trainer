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
        
        # Input channels for person image, clothing image, and person parse
        total_input_channels = input_channels_person + input_channels_cloth + input_channels_parse
        
        # Encoder
        self.enc1 = ConvBlock(total_input_channels, ngf)
        self.enc2 = DownsampleBlock(ngf, ngf*2)
        self.enc3 = DownsampleBlock(ngf*2, ngf*4)
        self.enc4 = DownsampleBlock(ngf*4, ngf*8)
        
        # Residual blocks
        self.res_blocks