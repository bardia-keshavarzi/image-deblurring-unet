"""
U‑Net for Image Deblurring
Resolution-preserving version — outputs same H×W as input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# Building blocks
# --------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.seq(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_ch, out_ch)
        )
    def forward(self, x):
        return self.mpconv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Fix potential off‑by‑1 pixel due to pooling rounding
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# --------------------------
# U-Net architecture
# --------------------------

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()

        # Encoder
        self.inc = ConvBlock(in_channels, base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bridge = ConvBlock(base_channels * 8, base_channels * 16)

        # Decoder
        self.up3 = UpBlock(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.up2 = UpBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.up1 = UpBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up0 = UpBlock(base_channels * 2 + base_channels, base_channels)

        # Output
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)       # -> 64
        x2 = self.down1(x1)    # -> 128
        x3 = self.down2(x2)    # -> 256
        x4 = self.down3(x3)    # -> 512

        # Bottleneck
        b = self.bridge(x4)

        # Decoder path (mirrors encoder)
        d3 = self.up3(b, x4)
        d2 = self.up2(d3, x3)
        d1 = self.up1(d2, x2)
        d0 = self.up0(d1, x1)

        out = self.outc(d0)
        return torch.sigmoid(out)


if __name__ == "__main__":
    model = UNet()
    x = torch.randn(1, 3, 384, 384)
    y = model(x)
    print("Input:", x.shape, "Output:", y.shape)
