"""
Loss functions for Image Deblurring
Includes Combined L1 + Perceptual Loss (safe for GPU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


# -------------------------------------------
# Perceptual (VGG16-based) Feature Loss
# -------------------------------------------

class PerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[3, 8, 15], weights=None):
        super().__init__()
        vgg = vgg16(weights='IMAGENET1K_V1').features.eval()
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:max(layer_ids) + 1])
        for p in self.vgg_layers.parameters():
            p.requires_grad = False

        self.layer_ids = layer_ids
        self.weights = weights if weights is not None else [1.0] * len(layer_ids)

        # Buffers auto-move with .to(device)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _to_device(self, device):
        """Move normalization and VGG layers to the selected device"""
        self.vgg_layers = self.vgg_layers.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def forward(self, pred, target):
        # Move VGG and buffers to the same device dynamically
        device = pred.device
        if next(self.vgg_layers.parameters()).device != device:
            self._to_device(device)

        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std

        loss = 0.0
        x, y = pred_norm, target_norm

        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            y = layer(y)
            if i in self.layer_ids:
                w = self.weights[self.layer_ids.index(i)]
                loss += w * F.l1_loss(x, y)

        return loss


# -------------------------------------------
# Combined Weighted Loss
# -------------------------------------------

class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=0.8, perceptual_weight=0.2):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss()

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        p_loss = self.perceptual(pred, target)
        total = self.l1_weight * l1_loss + self.perceptual_weight * p_loss
        return total


def get_loss_function():
    return CombinedLoss()
