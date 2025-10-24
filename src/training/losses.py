"""
Advanced Combined Loss for Image Deblurring
Includes:
- L1 loss
- Perceptual VGG16 feature loss
- SSIM loss
- Gradient/Edge loss (optional)
Optimized for mixed precision and GPU safety
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from torchmetrics.functional import structural_similarity_index_measure as ssim


# --------------------------------------------------
# Perceptual Loss (VGG16 Features)
# --------------------------------------------------
class PerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[3, 8, 15]):
        super().__init__()
        vgg = vgg16(weights="IMAGENET1K_V1").features.eval()
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:max(layer_ids) + 1])

        for p in self.vgg_layers.parameters():
            p.requires_grad = False

        self.layer_ids = layer_ids
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        device = pred.device
        self.vgg_layers = self.vgg_layers.to(device)

        # Normalize for VGG
        x = (pred - self.mean.to(device)) / self.std.to(device)
        y = (target - self.mean.to(device)) / self.std.to(device)

        loss = 0.0
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            y = layer(y)
            if i in self.layer_ids:
                loss += F.l1_loss(x, y)
        return loss / len(self.layer_ids)


# --------------------------------------------------
# Gradient (Edge) Loss
# --------------------------------------------------
def gradient_loss(pred, target):
    gx_pred = pred[:, :, :, :-1] - pred[:, :, :, 1:]
    gy_pred = pred[:, :, :-1, :] - pred[:, :, 1:, :]
    gx_targ = target[:, :, :, :-1] - target[:, :, :, 1:]
    gy_targ = target[:, :, :-1, :] - target[:, :, 1:, :]
    return F.l1_loss(gx_pred, gx_targ) + F.l1_loss(gy_pred, gy_targ)


# --------------------------------------------------
# Combined Weighted Loss
# --------------------------------------------------
class CombinedLoss(nn.Module):
    def __init__(self, l1_w=0.6, perc_w=0.25, ssim_w=0.1, grad_w=0.05):
        super().__init__()
        self.l1_w = l1_w
        self.perc_w = perc_w
        self.ssim_w = ssim_w
        self.grad_w = grad_w

        self.l1 = nn.L1Loss()
        self.perc = PerceptualLoss()

    def forward(self, pred, target):
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)

        l1_loss = self.l1(pred, target)
        perc_loss = self.perc(pred, target)
        ssim_loss = 1 - ssim(pred, target)
        grad_l = gradient_loss(pred, target)

        return (
            self.l1_w * l1_loss
            + self.perc_w * perc_loss
            + self.ssim_w * ssim_loss
            + self.grad_w * grad_l
        )


def get_loss_function():
    """Factory for trainer.py"""
    return CombinedLoss()