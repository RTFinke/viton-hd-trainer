#!/usr/bin/env python3
"""
Network components for VITON-HD
"""

import torch
import torch.nn as nn
import torchvision.models as models

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x + residual

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Generator(nn.Module):
    def __init__(self, input_channels=24, output_channels=3, ngf=64):
        super(Generator, self).__init__()
        
        self.enc1 = ConvBlock(input_channels, ngf)
        self.enc2 = DownsampleBlock(ngf, ngf*2)
        self.enc3 = DownsampleBlock(ngf*2, ngf*4)
        self.enc4 = DownsampleBlock(ngf*4, ngf*8)
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(ngf*8),
            ResidualBlock(ngf*8),
            ResidualBlock(ngf*8)
        )
        
        self.dec1 = UpsampleBlock(ngf*8, ngf*4)
        self.dec2 = UpsampleBlock(ngf*4, ngf*2)
        self.dec3 = UpsampleBlock(ngf*2, ngf)
        
        self.out_conv = nn.Conv2d(ngf, output_channels, kernel_size=3, padding=1)
        self.out_activation = nn.Tanh()
    
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        
        x_res = self.res_blocks(x4)
        
        x = self.dec1(x_res)
        x = self.dec2(x)
        x = self.dec3(x)
        
        x = self.out_conv(x)
        x = self.out_activation(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channels=6, ndf=64):
        super(Discriminator, self).__init__()
        
        self.layers = nn.Sequential(
            ConvBlock(input_channels, ndf, kernel_size=4, padding=1),
            DownsampleBlock(ndf, ndf*2),
            DownsampleBlock(ndf*2, ndf*4),
            DownsampleBlock(ndf*4, ndf*8),
            nn.Conv2d(ndf*8, 1, kernel_size=4, padding=1)
        )
    
    def forward(self, x):
        return self.layers(x)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features)[:16])  # Up to conv4_3
    
    def forward(self, x):
        return self.features(x)

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.criterion = nn.L1Loss()
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, generated, target):
        gen_features = self.feature_extractor(generated)
        target_features = self.feature_extractor(target)
        return self.criterion(gen_features, target_features)