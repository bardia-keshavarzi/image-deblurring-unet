"""
Residual U-Net with optional attention for Image Deblurring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# Basic building blocks
# --------------------------

class ConvBlock(nn.Module):
    """Residual convolutional block"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.res = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        identity = x if self.res is None else self.res(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_ch, out_ch))
    def forward(self, x): return self.down(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], 1)
        return self.conv(x)


# --------------------------
# U-Net (Residual)
# --------------------------

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base=64):
        super().__init__()

        self.inc = ConvBlock(in_channels, base)
        self.down1 = DownBlock(base, base * 2)
        self.down2 = DownBlock(base * 2, base * 4)
        self.down3 = DownBlock(base * 4, base * 8)
        self.bridge = ConvBlock(base * 8, base * 16)
        self.up3 = UpBlock(base * 8 + base * 16, base * 8)
        self.up2 = UpBlock(base * 8 + base * 4, base * 4)
        self.up1 = UpBlock(base * 4 + base * 2, base * 2)
        self.up0 = UpBlock(base * 2 + base, base)
        self.outc = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        b = self.bridge(x4)
        d3 = self.up3(b, x4)
        d2 = self.up2(d3, x3)
        d1 = self.up1(d2, x2)
        d0 = self.up0(d1, x1)
        return torch.sigmoid(self.outc(d0))
