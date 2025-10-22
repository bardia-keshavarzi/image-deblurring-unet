# src/models/unet.py
"""
Simple U-Net for Image Deblurring

U-Net Architecture:
- Encoder: 4 downsampling blocks (256→128→64→32→16)
- Bottleneck: Deep features
- Decoder: 4 upsampling blocks (16→32→64→128→256)
- Skip connections: Preserve details

Input: Blurred RGB image (3, 256, 256)
Output: Sharp RGB image (3, 256, 256)
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two consecutive Conv2d + BatchNorm + ReLU
    
    This is the basic building block used throughout U-Net
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """
    Downsampling block: MaxPool + DoubleConv
    
    Reduces spatial size by 2x, increases channels
    Example: (64, 128, 128) → (128, 64, 64)
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """
    Upsampling block: Upsample + Concatenate + DoubleConv
    
    Increases spatial size by 2x, decreases channels
    Uses skip connections from encoder
    Example: (128, 64, 64) + skip → (64, 128, 128)
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: upsampled features from decoder
            x2: skip connection from encoder
        """
        x1 = self.up(x1)
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Simple U-Net for Image Deblurring
    
    Architecture:
        Encoder (downsampling):
        - Block 1: 3   → 64  (256×256)
        - Block 2: 64  → 128 (128×128)
        - Block 3: 128 → 256 (64×64)
        - Block 4: 256 → 512 (32×32)
        
        Bottleneck:
        - Block 5: 512 → 1024 (16×16)
        
        Decoder (upsampling with skip connections):
        - Block 6: 1024 → 512 (32×32)
        - Block 7: 512  → 256 (64×64)
        - Block 8: 256  → 128 (128×128)
        - Block 9: 128  → 64  (256×256)
        
        Output: 64 → 3 (RGB)
    
    Parameters: ~7.8M
    """
    
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder (downsampling)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # Bottleneck
        self.down4 = Down(512, 1024)
        
        # Decoder (upsampling)
        self.up1 = Up(1024 + 512, 512)  # 1024 from bottleneck + 512 skip
        self.up2 = Up(512 + 256, 256)   # 512 from up1 + 256 skip
        self.up3 = Up(256 + 128, 128)   # 256 from up2 + 128 skip
        self.up4 = Up(128 + 64, 64)     # 128 from up3 + 64 skip
        
        # Output layer
        self.outc = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return torch.sigmoid(out)  



def count_parameters(model):
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += p.numel()
    return total
