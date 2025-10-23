# src/training/losses.py
"""
Loss Functions - IMPROVED

Adds perceptual loss for better visual quality
Also enforces size alignment and device consistency to avoid runtime errors.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss with device-safe buffers"""
    
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16]  # Up to relu3_3
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        # Register on-module buffers; they'll move with .to(device)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        # Ensure VGG on the same device as inputs
        if next(self.vgg.parameters()).device != pred.device:
            self.vgg = self.vgg.to(pred.device)
        # Expect both in [-1,1]; map to [0,1] before ImageNet norm
        pred_01 = (pred + 1.0) * 0.5
        target_01 = (target + 1.0) * 0.5
        pred_norm = (pred_01 - self.mean) / self.std
        target_norm = (target_01 - self.mean) / self.std
        pred_features = self.vgg(pred_norm)
        target_features = self.vgg(target_norm)
        return self.l1(pred_features, target_features)


class CombinedLoss(nn.Module):
    """L1 + Perceptual Loss with shape/device-safe alignment"""
    
    def __init__(self, alpha=0.8, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss()
    
    @staticmethod
    def _match_size(x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=True)
            
        return x
    
    def forward(self, pred, target):
        # Move perceptual submodules/buffers to the correct device if needed
        if self.perceptual.mean.device != pred.device:
            self.perceptual.to(pred.device)
        # Ensure same spatial size
        pred = self._match_size(pred, target)
        # L1 in same scale
        l1_loss = self.l1(pred, target)
        # Perceptual
        perceptual_loss = self.perceptual(pred, target)
        return self.alpha * l1_loss + self.beta * perceptual_loss


def get_loss_function():
    return CombinedLoss(alpha=0.8, beta=0.2)
