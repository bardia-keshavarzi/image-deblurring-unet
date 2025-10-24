"""
Residual + SE-Attention U-Net for Image Deblurring
Faster convergence, sharper restorations, and higher PSNR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------
# SE Attention Block
# ----------------------------------
class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.fc(w)
        return x * w


# ----------------------------------
# Residual Conv Block (Improved)
# ----------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_attn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.res = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.attn = SEBlock(out_ch) if use_attn else nn.Identity()

    def forward(self, x):
        identity = x if self.res is None else self.res(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.attn(out)
        return F.relu(out)


# ----------------------------------
# Down / Up Blocks
# ----------------------------------
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_attn=False):
        super().__init__()
        self.down = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_ch, out_ch, use_attn))
    def forward(self, x): return self.down(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_attn=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch, out_ch, use_attn)
    def forward(self, x, skip):
        x = self.up(x)
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ----------------------------------
# Full U-Net
# ----------------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base=64, use_attention=True):
        super().__init__()
        self.inc = ConvBlock(in_channels, base, use_attention)
        self.down1 = DownBlock(base, base * 2, use_attention)
        self.down2 = DownBlock(base * 2, base * 4, use_attention)
        self.down3 = DownBlock(base * 4, base * 8, use_attention)

        self.bridge = ConvBlock(base * 8, base * 16, use_attention)

        self.up3 = UpBlock(base * 16 + base * 8, base * 8, use_attention)
        self.up2 = UpBlock(base * 8 + base * 4, base * 4, use_attention)
        self.up1 = UpBlock(base * 4 + base * 2, base * 2, use_attention)
        self.up0 = UpBlock(base * 2 + base, base, use_attention)

        self.outc = nn.Conv2d(base, out_channels, kernel_size=1)

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
