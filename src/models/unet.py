import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        max_ = self.fc(self.max_pool(x))
        return self.sigmoid(avg + max_) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg, max_], dim=1)
        return self.sigmoid(self.conv(out)) * x


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_attn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.res = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.attn = CBAMBlock(out_ch) if use_attn else nn.Identity()

    def forward(self, x):
        ident = self.res(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.attn(out + ident)
        return F.relu(out, inplace=True)



class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_attn=False):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ResBlock(in_ch, out_ch, use_attn)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_attn=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ResBlock(in_ch, out_ch, use_attn)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        return self.conv(torch.cat([x, skip], dim=1))


# ===========================================================
# Multi-Scale Residual U-Net
# ===========================================================
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base=64, use_attention=True):
        super().__init__()

        # Encoder
        self.inc = ResBlock(in_channels, base, use_attn=use_attention)
        self.down1 = DownBlock(base, base * 2, use_attn=use_attention)
        self.down2 = DownBlock(base * 2, base * 4, use_attn=use_attention)
        self.down3 = DownBlock(base * 4, base * 8, use_attn=use_attention)
        self.down4 = DownBlock(base * 8, base * 16, use_attn=use_attention)

        # Bridge
        self.bridge = nn.Sequential(
            ResBlock(base * 16, base * 32, use_attn=True),
            CBAMBlock(base * 32)
        )

        # Decoder
        self.up4 = UpBlock(base * 32 + base * 16, base * 16, use_attn=use_attention)
        self.up3 = UpBlock(base * 16 + base * 8, base * 8, use_attn=use_attention)
        self.up2 = UpBlock(base * 8 + base * 4, base * 4, use_attn=use_attention)
        self.up1 = UpBlock(base * 4 + base * 2, base * 2, use_attn=use_attention)
        self.up0 = UpBlock(base * 2 + base, base, use_attn=use_attention)

        self.outc = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        b = self.bridge(x5)

        d4 = self.up4(b, x5)
        d3 = self.up3(d4, x4)
        d2 = self.up2(d3, x3)
        d1 = self.up1(d2, x2)
        d0 = self.up0(d1, x1)
        return torch.sigmoid(self.outc(d0))
