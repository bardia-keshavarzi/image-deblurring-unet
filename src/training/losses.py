"""
Combined L1 + Perceptual Loss (GPU‑safe)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


# ------------------------------
# Perceptual (VGG16) Feature Loss
# ------------------------------
class PerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[3, 8, 15]):
        super().__init__()
        vgg = vgg16(weights="IMAGENET1K_V1").features.eval()
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:max(layer_ids) + 1])
        for p in self.vgg_layers.parameters():
            p.requires_grad = False

        self.layer_ids = layer_ids
        # Buffers that track normalization and move automatically with .to(device)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        device = pred.device
        # ensure VGG and buffers are on the same device as inputs
        self.vgg_layers = self.vgg_layers.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

        x = (pred - self.mean) / self.std
        y = (target - self.mean) / self.std

        loss = 0.0
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            y = layer(y)
            if i in self.layer_ids:
                loss += F.l1_loss(x, y)
        return loss / len(self.layer_ids)


# ------------------------------
# Combined Weighted Loss
# ------------------------------
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
        return self.l1_weight * l1_loss + self.perceptual_weight * p_loss


def get_loss_function():
    return CombinedLoss()
