"""
Loss functions for Image Deblurring
Includes combined L1 + Perceptual loss with safe device placement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


# -----------------------
# Perceptual (VGG‑based)
# -----------------------

class PerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[3, 8, 15], weights=None):
        super().__init__()
        vgg = vgg16(weights='IMAGENET1K_V1').features.eval()
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:max(layer_ids) + 1])
        for p in self.vgg_layers.parameters():
            p.requires_grad = False

        self.layer_ids = layer_ids
        self.weights = weights if weights is not None else [1.0] * len(layer_ids)

        # Normalization constants (will move to same device later)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        # Ensure mean/std are on the same device
        mean, std = self.mean.to(pred.device), self.std.to(pred.device)

        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std

        loss = 0.0
        x, y = pred_norm, target_norm

        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            y = layer(y)
            if i in self.layer_ids:
                loss += self.weights[self.layer_ids.index(i)] * F.l1_loss(x, y)

        return loss


# -----------------------
# Combined Loss
# -----------------------

class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=0.8, perceptual_weight=0.2):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss()

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        perceptual_loss = self.perceptual(pred, target)
        return self.l1_weight * l1_loss + self.perceptual_weight * perceptual_loss


# -----------------------
# Factory Method
# -----------------------

def get_loss_function():
    return CombinedLoss()
